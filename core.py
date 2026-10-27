"""Forecasting logic with no Streamlit dependency.

Shared by app.py (the UI) and weekly_report.py (scheduled emails).

- ONE feature function (`build_features`) is used for training, validation and the live
  forecast, so the model never sees a differently-defined feature at prediction time.
- History lives on a full daily calendar; closed/missing days are NaN.
- Metrics are honest: WAPE, a 7-day-ahead recursive back-test, and a naive baseline.
- Forecast ranges come from the back-test's real 7-day errors (empirical quantiles).
"""
import io
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

MIN_DAYS = 60            # minimum rows of valid history
MIN_TRAIN_ROWS = 40      # minimum rows left after feature engineering
LOOKBACK_DAYS = 60       # history window needed to compute features for one day
HORIZON = 7              # planner length in days
TEST_FRACTION = 0.2      # chronological hold-out
CLOSED_THRESHOLD = 0.2   # weekday seen on < 20% of its calendar dates => regularly closed
DEFAULT_TEMP = 15.0      # neutral weather when no weather data is available
MAX_SPAN_DAYS = 365 * 15
INTERVAL_LOW = 10        # forecast range = 10th..90th percentile of past errors (80% range)
INTERVAL_HIGH = 90
INTERVAL_MIN_SAMPLES = 15

FEATURES = [
    "day_of_week", "is_weekend", "month", "day_of_month", "is_payday",
    "is_holiday", "is_holiday_tomorrow", "is_school_break",
    "temp", "rain", "rain_weekend_penalty", "is_patio_weather",
    "visitors_yesterday", "visitors_7d_avg", "visitors_30d_avg",
    "visitors_same_day_last_week", "post_closure_surge",
    "visitors_7d_max", "visitors_30d_max", "momentum_ratio",
]


# ----------------------------------------------------------------------------
# Data loading & validation
# ----------------------------------------------------------------------------
def read_csv_bytes(raw_bytes):
    """Parse an uploaded CSV. Handles BOMs, Latin-1 files and ; or , or tab separators."""
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


DATE_ALIASES = {"visit_date", "date", "business_date", "day", "datum", "transaction_date", "order_date"}
VISITOR_ALIASES = {"visitors", "covers", "guests", "pax", "headcount", "customers", "client_count", "anzahl"}


def _standardize_pos_columns(raw_df):
    """Map common POS export column names to 'visit_date' and 'visitors'."""
    df = raw_df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    col_map = {}
    if "visit_date" not in df.columns:
        for col in df.columns:
            if col in DATE_ALIASES:
                col_map[col] = "visit_date"
                break
    if "visitors" not in df.columns:
        for col in df.columns:
            if col in VISITOR_ALIASES:
                col_map[col] = "visitors"
                break
    return df.rename(columns=col_map) if col_map else df


def clean_visits(raw_df, min_days=MIN_DAYS):
    """Validate and clean an upload.

    Returns (clean_df, notes, error). `notes` is a list of (level, message_key, kwargs);
    `error` is None or (message_key, kwargs). Messages are rendered by the UI (localized).
    """
    raw_df = _standardize_pos_columns(raw_df)
    if "visit_date" not in raw_df.columns or "visitors" not in raw_df.columns:
        return None, [], ("err_columns", {})

    notes = []
    dates_raw = raw_df["visit_date"].astype(str).str.strip()
    if dates_raw.str.match(r"^\d{4}-\d{2}-\d{2}").all():
        parsed = pd.to_datetime(dates_raw, errors="coerce")
    else:
        # Non-ISO dates: assume European DD/MM/YYYY and tell the user.
        parsed = pd.to_datetime(dates_raw, errors="coerce", dayfirst=True)
        notes.append(("info", "info_dayfirst", {}))
    parsed = parsed.dt.normalize()
    visitors = pd.to_numeric(raw_df["visitors"], errors="coerce")

    valid = parsed.notna() & visitors.notna() & (visitors >= 0)
    if not valid.any():
        return None, notes, ("err_no_valid_rows", {})
    dropped = int((~valid).sum())
    if dropped:
        notes.append(("warning", "warn_dropped", {"n": dropped}))

    clean = pd.DataFrame({"visit_date": parsed[valid], "visitors": visitors[valid].astype(float)})
    duplicates = int(clean["visit_date"].duplicated().sum())
    if duplicates:
        notes.append(("info", "warn_duplicates", {"n": duplicates}))
    clean = clean.groupby("visit_date", as_index=False)["visitors"].sum().sort_values("visit_date")

    # Treat explicit 0-visitor days from POS exports as closed days (absent in `visits` -> NaN on calendar)
    clean = clean[clean["visitors"] > 0].reset_index(drop=True)
    if clean.empty:
        return None, notes, ("err_no_valid_rows", {})

    if (clean["visit_date"].max() - clean["visit_date"].min()).days > MAX_SPAN_DAYS:
        return None, notes, ("err_dates", {})
    if len(clean) < min_days:
        return None, notes, ("err_too_short", {"min": min_days, "found": len(clean)})
    return clean, notes, None


CAPITAL_COORDS = {
    "FR": (48.8566, 2.3522),
    "DE": (52.5200, 13.4050),
    "AT": (48.2082, 16.3738),
    "CH": (47.3769, 8.5417),
    "BE": (50.8503, 4.3517),
    "IT": (41.9028, 12.4964),
    "ES": (40.4168, -3.7038),
    "NL": (52.3676, 4.9041),
    "GB": (51.5074, -0.1278),
}


def resolve_location_coords(location_query, country="FR"):
    """Resolve a city name or postal code to (lat, lon) via Open-Meteo Geocoding API.
    Falls back to the country's capital coordinates if the query is empty or a legacy numeric ID."""
    query = str(location_query or "").strip()
    if not query or query.isdigit():
        return CAPITAL_COORDS.get(country, CAPITAL_COORDS["FR"])

    for use_country in (True, False):
        params = {"name": query, "count": 1, "language": "en", "format": "json"}
        if use_country and country:
            params["countryCode"] = country
        url = f"https://geocoding-api.open-meteo.com/v1/search?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            results = payload.get("results")
            if results:
                return float(results[0]["latitude"]), float(results[0]["longitude"])
        except Exception:
            break
    return CAPITAL_COORDS.get(country, CAPITAL_COORDS["FR"])


def _fetch_open_meteo_range(lat, lon, start_iso, end_iso):
    """Fetch daily temperature_2m_mean and precipitation_sum from Open-Meteo Archive & Forecast APIs."""
    start_d = date.fromisoformat(start_iso)
    end_d = date.fromisoformat(end_iso)
    today_d = date.today()
    max_forecast_d = today_d + timedelta(days=15)
    if start_d > max_forecast_d:
        return None

    cutoff_d = today_d - timedelta(days=5)
    frames = []

    def _call_api(base_url, s_date, e_date):
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": s_date.isoformat(),
            "end_date": e_date.isoformat(),
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "auto",
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            daily_data = json.loads(resp.read().decode("utf-8")).get("daily", {})
        times = daily_data.get("time", [])
        if not times:
            return None
        df = pd.DataFrame({
            "date": pd.to_datetime(times).normalize(),
            "temp": pd.to_numeric(daily_data.get("temperature_2m_mean", [np.nan] * len(times)), errors="coerce"),
            "rain": pd.to_numeric(daily_data.get("precipitation_sum", [np.nan] * len(times)), errors="coerce"),
        })
        return df

    if start_d <= cutoff_d:
        arch_end = min(end_d, cutoff_d)
        df_arch = _call_api("https://archive-api.open-meteo.com/v1/archive", start_d, arch_end)
        if df_arch is not None:
            frames.append(df_arch)

    if end_d > cutoff_d:
        fc_start = max(start_d, cutoff_d + timedelta(days=1))
        fc_end = min(end_d, max_forecast_d)
        if fc_start <= fc_end:
            df_fc = _call_api("https://api.open-meteo.com/v1/forecast", fc_start, fc_end)
            if df_fc is not None:
                frames.append(df_fc)

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last")
    return combined.set_index("date").sort_index()


def fetch_history_weather(station, start_iso, end_iso, country="FR"):
    """Daily historical weather from Open-Meteo (used as a model feature).
    Returns (DataFrame[temp, rain] indexed by date, None) or (None, reason)."""
    try:
        lat, lon = resolve_location_coords(station, country)
        out = _fetch_open_meteo_range(lat, lon, start_iso, end_iso)
        if out is not None and len(out) > 0:
            return out, None
        return None, "no data"
    except Exception as exc:
        return None, type(exc).__name__


def fetch_future_weather(station, country, future_dates, history):
    """Fetch day-by-day weather forecast from Open-Meteo for `future_dates`.
    Falls back to `default_weather(history)` for any missing dates.
    Returns (DataFrame[temp, rain] indexed by future_dates, live_weather_ok)."""
    def_temp, def_rain = default_weather(history)
    fallback = pd.DataFrame({"temp": def_temp, "rain": def_rain}, index=future_dates)
    if len(future_dates) == 0:
        return fallback, False
    try:
        lat, lon = resolve_location_coords(station, country)
        raw = _fetch_open_meteo_range(
            lat, lon, future_dates[0].date().isoformat(), future_dates[-1].date().isoformat()
        )
        if raw is None or raw.empty:
            return fallback, False
        reindexed = raw.reindex(future_dates)
        live_ok = bool(reindexed["temp"].notna().any())
        reindexed["temp"] = reindexed["temp"].fillna(def_temp).round(1)
        reindexed["rain"] = reindexed["rain"].fillna(def_rain).round(1)
        return reindexed[["temp", "rain"]], live_ok
    except Exception:
        return fallback, False


# ----------------------------------------------------------------------------
# Features (single source of truth for training AND forecasting)
# ----------------------------------------------------------------------------
def school_break_flags(idx):
    """Rough school-holiday / tourism-season flag. Dates are approximations, not an official calendar."""
    m, d = idx.month.to_numpy(), idx.day.to_numpy()
    flag = (
        ((m == 4) & (d >= 10) & (d <= 25))
        | np.isin(m, [7, 8])
        | ((m == 12) & (d >= 20))
        | ((m == 1) & (d <= 5))
        | ((m == 10) & (d >= 25))
    )
    return flag.astype(int)


def get_holiday_dates(country, calendar_index):
    years = list(range(int(calendar_index.year.min()), int(calendar_index.year.max()) + 2))
    parts = str(country or "FR").split("-", 1)
    base_country = parts[0].strip()
    subdiv = parts[1].strip() if len(parts) > 1 else None
    try:
        hol = holidays.country_holidays(base_country, subdiv=subdiv, years=years)
    except Exception:
        hol = holidays.country_holidays(base_country, years=years)
    return pd.DatetimeIndex(pd.to_datetime(list(hol.keys())))


def build_features(frame, holiday_dates):
    """frame: daily-calendar DataFrame with columns visitors (NaN = closed/unknown), temp, rain.
    Every lag uses only data from BEFORE the row's own day."""
    idx = frame.index
    out = pd.DataFrame(index=idx)

    dow = idx.dayofweek.to_numpy()
    day = idx.day.to_numpy()
    days_in_month = idx.days_in_month.to_numpy()
    out["day_of_week"] = dow
    out["is_weekend"] = (dow >= 5).astype(int)
    out["month"] = idx.month.to_numpy()
    out["day_of_month"] = day
    # Includes European end-of-month salary dates (25th through 1st) plus mid-month (15th)
    out["is_payday"] = ((day == 1) | (day == 15) | (day >= 25) | (day >= days_in_month - 2)).astype(int)
    out["is_holiday"] = idx.isin(holiday_dates).astype(int)
    out["is_holiday_tomorrow"] = (idx + pd.Timedelta(days=1)).isin(holiday_dates).astype(int)
    out["is_school_break"] = school_break_flags(idx)
    out["temp"] = frame["temp"].to_numpy()
    out["rain"] = frame["rain"].to_numpy()
    out["rain_weekend_penalty"] = out["rain"] * out["is_weekend"]
    out["is_patio_weather"] = ((out["temp"] >= 18.0) & (out["rain"] <= 0.5)).astype(int)

    visitors = frame["visitors"]
    past = visitors.shift(1)  # strictly earlier days only
    avg7 = past.rolling(7, min_periods=3).mean()
    avg30 = past.rolling(30, min_periods=10).mean()
    out["visitors_7d_avg"] = avg7
    out["visitors_30d_avg"] = avg30
    out["visitors_7d_max"] = past.rolling(7, min_periods=3).max()
    out["visitors_30d_max"] = past.rolling(30, min_periods=10).max()
    out["post_closure_surge"] = past.isna().astype(int)
    # If yesterday / last week was a closed day, fall back to the recent average.
    out["visitors_yesterday"] = past.fillna(avg7)
    out["visitors_same_day_last_week"] = visitors.shift(7).fillna(avg7)
    out["momentum_ratio"] = (avg7 / avg30.replace(0, np.nan)).fillna(1.0).clip(0.5, 2.0)
    return out


def detect_closed_weekdays(visits, calendar_index):
    observed = visits["visit_date"].dt.dayofweek.value_counts()
    expected = pd.Series(calendar_index.dayofweek).value_counts()
    closed = {
        d for d in range(7)
        if expected.get(d, 0) > 0 and observed.get(d, 0) / expected[d] < CLOSED_THRESHOLD
    }
    return frozenset() if len(closed) >= 7 else frozenset(closed)


def wape(actual, predicted):
    """Weighted absolute percentage error: sum|error| / sum(actual), in %."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    total = actual.sum()
    return float(np.abs(actual - predicted).sum() / total * 100) if total > 0 else float("nan")


def fmt_pct(value):
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.1f}%"


# ----------------------------------------------------------------------------
# Model, validation, forecast ranges
# ----------------------------------------------------------------------------
def fit_forest(X, y):
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    model.set_params(n_jobs=1)  # single-row predictions are faster without thread overhead
    return model


@dataclass
class Forecaster:
    model: RandomForestRegressor
    holiday_dates: pd.DatetimeIndex
    closed_weekdays: frozenset
    medians: pd.Series

    def forecast(self, history, future_weather):
        """Recursive multi-day forecast.

        history: daily DataFrame (visitors, temp, rain), last LOOKBACK_DAYS or so.
        future_weather: DataFrame[temp, rain] indexed by the future dates.
        Returns a Series of predictions (NaN on regularly closed weekdays). Each prediction is
        written back into the history so the next day's lag features use it.
        """
        frame = history[["visitors", "temp", "rain"]].copy()
        preds = {}
        for d, w in future_weather.iterrows():
            frame.loc[d] = [np.nan, float(w["temp"]), float(w["rain"])]
            if d.dayofweek in self.closed_weekdays:
                preds[d] = np.nan
                continue
            feats = build_features(frame.tail(LOOKBACK_DAYS), self.holiday_dates)
            X = feats[FEATURES].iloc[[-1]].fillna(self.medians)
            pred = max(0.0, float(self.model.predict(X)[0]))
            frame.loc[d, "visitors"] = pred
            preds[d] = pred
        return pd.Series(preds, dtype=float)


def validate_seven_day(forecaster, history, features, first_test_date):
    """Back-test the planner itself: from many origins in the test period, forecast 7 days
    recursively (with that period's real weather) and score against what really happened.

    Returns a dict: wape7, base_wape7 (in %), and ratios_by_h = {h: [actual/predicted, ...]}
    which feed the forecast ranges."""
    step = pd.Timedelta(days=HORIZON)
    last_date = history.index[-1]
    origin = first_test_date - pd.Timedelta(days=1)
    actual_all, model_all, base_all = [], [], []
    ratios = {h: [] for h in range(1, HORIZON + 1)}
    while origin + step <= last_date:
        future = pd.date_range(origin + pd.Timedelta(days=1), periods=HORIZON, freq="D")
        preds = forecaster.forecast(
            history.loc[:origin].tail(LOOKBACK_DAYS), history.loc[future, ["temp", "rain"]]
        ).reindex(future)
        actual = history.loc[future, "visitors"]
        base = features.loc[future, "visitors_same_day_last_week"]
        ok = (actual.notna() & preds.notna() & base.notna()).to_numpy()
        actual_all.append(actual.to_numpy()[ok])
        model_all.append(preds.to_numpy()[ok])
        base_all.append(base.to_numpy()[ok])
        pred_arr, act_arr = preds.to_numpy(), actual.to_numpy()
        for i in range(HORIZON):
            if np.isfinite(pred_arr[i]) and np.isfinite(act_arr[i]) and pred_arr[i] > 0:
                ratios[i + 1].append(float(act_arr[i] / pred_arr[i]))
        origin += step
    if not actual_all:
        return {"wape7": float("nan"), "base_wape7": float("nan"), "ratios_by_h": ratios}
    a = np.concatenate(actual_all)
    return {
        "wape7": wape(a, np.concatenate(model_all)),
        "base_wape7": wape(a, np.concatenate(base_all)),
        "ratios_by_h": ratios,
    }


def compute_intervals(ratios_by_h):
    """Turn past actual/predicted ratios into {h: (low_ratio, high_ratio)} for h = 1..HORIZON.
    Uses that horizon's own errors when there are enough, otherwise all horizons pooled.
    Returns None when there is too little back-test data for a trustworthy range."""
    arrays = [np.asarray(v, dtype=float) for v in ratios_by_h.values() if len(v)]
    pooled = np.concatenate(arrays) if arrays else np.array([])
    if len(pooled) < INTERVAL_MIN_SAMPLES:
        return None
    pooled_q = np.percentile(pooled, [INTERVAL_LOW, INTERVAL_HIGH])
    out = {}
    for h in range(1, HORIZON + 1):
        values = np.asarray(ratios_by_h.get(h, []), dtype=float)
        q = np.percentile(values, [INTERVAL_LOW, INTERVAL_HIGH]) if len(values) >= INTERVAL_MIN_SAMPLES else pooled_q
        out[h] = (float(q[0]), float(q[1]))
    return out


def train_bundle(visits, station, country, events=None):
    """Everything expensive, computed once per (data, station, country, events).
    Returns None when there is not enough usable history."""
    calendar = pd.date_range(visits["visit_date"].min(), visits["visit_date"].max(), freq="D")
    raw_visitors = visits.set_index("visit_date")["visitors"].reindex(calendar)
    if events:
        extra_series = pd.Series(
            [float(events.get(d.strftime("%Y-%m-%d"), 0)) for d in calendar],
            index=calendar,
            dtype=float,
        )
        # Subtract one-off private event spikes so lags and training reflect organic traffic
        adjusted = np.where(
            raw_visitors.notna(),
            np.maximum(1.0, raw_visitors - extra_series),
            np.nan,
        )
        raw_visitors = pd.Series(adjusted, index=calendar)
    history = pd.DataFrame({"visitors": raw_visitors})

    weather, weather_reason = fetch_history_weather(
        station, calendar[0].date().isoformat(), calendar[-1].date().isoformat(), country=country
    )
    weather_ok = weather is not None and bool(weather["temp"].notna().any() or weather["rain"].notna().any())
    if weather_ok:
        temp = weather["temp"].reindex(calendar)
        rain = weather["rain"].reindex(calendar)
        history["temp"] = temp.fillna(temp.mean() if temp.notna().any() else DEFAULT_TEMP)
        history["rain"] = rain.fillna(0.0)
    else:
        history["temp"] = DEFAULT_TEMP
        history["rain"] = 0.0

    holiday_dates = get_holiday_dates(country, calendar)
    closed = detect_closed_weekdays(visits, calendar)
    features = build_features(history, holiday_dates)

    usable = history["visitors"].notna() & features[FEATURES].notna().all(axis=1)
    X, y = features.loc[usable, FEATURES], history.loc[usable, "visitors"]
    if len(X) < MIN_TRAIN_ROWS:
        return None

    # 1) Honest validation: train on the first 80%, test on the most recent 20%.
    n_train = int(len(X) * (1 - TEST_FRACTION))
    X_train, X_test, y_train, y_test = X.iloc[:n_train], X.iloc[n_train:], y.iloc[:n_train], y.iloc[n_train:]
    val_model = fit_forest(X_train, y_train)
    val_forecaster = Forecaster(val_model, holiday_dates, closed, X_train.median())

    preds = val_model.predict(X_test)
    baseline = X_test["visitors_same_day_last_week"].to_numpy()
    validation = validate_seven_day(val_forecaster, history, features, X_test.index[0])

    n_chart = min(30, len(X_test))
    chart = pd.DataFrame(
        {"actual": y_test.to_numpy()[-n_chart:], "pred": preds[-n_chart:], "baseline": baseline[-n_chart:]},
        index=X_test.index[-n_chart:],
    )

    # 2) The model used for the planner is refit on ALL the data (incl. the most recent period).
    final_model = fit_forest(X, y)
    dow_means = visits.groupby(visits["visit_date"].dt.dayofweek)["visitors"].mean().reindex(range(7)).fillna(0.0)

    return {
        "forecaster": Forecaster(final_model, holiday_dates, closed, X.median()),
        "history": history,
        "closed_weekdays": closed,
        "mean_visitors": float(visits["visitors"].mean()),
        "mae": float(mean_absolute_error(y_test, preds)),
        "wape": wape(y_test, preds),
        "base_wape": wape(y_test, baseline),
        "wape7": validation["wape7"],
        "base_wape7": validation["base_wape7"],
        "intervals": compute_intervals(validation["ratios_by_h"]),
        "chart": chart,
        "dow_means": dow_means,
        "weather_ok": weather_ok,
        "weather_reason": weather_reason,
    }


# ----------------------------------------------------------------------------
# Planning helpers (shared by the app and the weekly email)
# ----------------------------------------------------------------------------
def future_dates_after(history, days=HORIZON):
    return pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")


def default_weather(history):
    """Neutral weather assumption: the last week's average temperature, no rain."""
    return float(np.clip(round(history["temp"].tail(7).mean()), -10, 40)), 0.0


def build_plan(base_preds, intervals, last_date, event_date, event_extra,lunch_share, avg_spend, guests_per_server, events=None):
    """Turn raw model predictions into the planner rows (numbers only, no formatting).

    Each row: date, expected, low, high (None without intervals), is_closed, on_event,
    lunch, dinner, lunch_servers, dinner_servers, servers, revenue.
    """
    plan = []
    events_map = events or {}
    for d, base in base_preds.items():
        d_iso = d.strftime("%Y-%m-%d")
        saved_extra = int(events_map.get(d_iso, 0))
        single_extra = int(event_extra) if (event_date is not None and d.date() == event_date) else 0
        extra = single_extra if single_extra > 0 else saved_extra  # one-off bump, never fed back into lags
        on_event = extra > 0
        regularly_closed = bool(pd.isna(base))
        is_closed = regularly_closed and not on_event
        expected = extra if regularly_closed else int(round(base)) + extra

        low = high = None
        if intervals is not None:
            if regularly_closed:
                low = high = expected
            else:
                h = min(HORIZON, max(1, (d - last_date).days))
                q_lo, q_hi = intervals[h]
                low = min(expected, int(round(base * q_lo)) + extra)
                high = max(expected, int(round(base * q_hi)) + extra)

        if is_closed or expected == 0:
            lunch = dinner = lunch_servers = dinner_servers = servers = 0
        else:
            lunch = int(round(expected * lunch_share / 100))
            dinner = expected - lunch
            lunch_servers = math.ceil(lunch / guests_per_server) if lunch > 0 else 0
            dinner_servers = math.ceil(dinner / guests_per_server) if dinner > 0 else 0
            servers = max(1, lunch_servers + dinner_servers)

        plan.append({
            "date": d, "expected": expected, "low": low, "high": high,
            "is_closed": is_closed, "on_event": on_event,
            "lunch": lunch, "dinner": dinner,
            "lunch_servers": lunch_servers, "dinner_servers": dinner_servers,
            "servers": servers,
            "revenue": round(expected * avg_spend, 2),
        })
    return plan
