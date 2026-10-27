"""AI Business Forecaster (Streamlit UI).

The forecasting logic lives in core.py, files/history in storage.py, PDF + email in reports.py,
text in i18n.py. See README.md for setup, the weekly email and the password.
"""
import csv
import hmac
import os
from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import core
import mailer
import reports
import storage
from i18n import COUNTRY_CURRENCIES, COUNTRY_NAMES, DAY_TRANSLATIONS, MONTH_TRANSLATIONS, TRANSLATIONS

st.set_page_config(page_title="AI Traffic Forecaster", layout="wide")


# ----------------------------------------------------------------------------
# Cached wrappers (so moving a widget never retrains the model)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_read_csv(raw_bytes):
    return core.read_csv_bytes(raw_bytes)


@st.cache_resource(show_spinner=False)
def cached_train(visits, station, country, events=None):
    return core.train_bundle(visits, station, country, events=events)


def get_secret(name):
    """Environment variable first, then Streamlit secrets (if a secrets file exists)."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets.toml configured
        return None


# ============================================================================
# Page setup, language, optional password
# ============================================================================
for key, default in (
    ("ai_trained", False), ("file_sig", None), ("authed", False),
    ("auth_role", "admin"), ("locked_slug", None), ("cloud_pulled", False),
):
    if key not in st.session_state:
        st.session_state[key] = default

if not st.session_state.cloud_pulled:
    storage.pull_all_from_cloud()
    st.session_state.cloud_pulled = True

st.sidebar.subheader("🌐 Language / Langue / Sprache")
selected_lang = st.sidebar.selectbox(
    "Choose Language", ["English", "French", "German"], key="language_selector", label_visibility="collapsed"
)
t = TRANSLATIONS[selected_lang]
DAYS = DAY_TRANSLATIONS[selected_lang]
MONTHS = MONTH_TRANSLATIONS[selected_lang]

st.title(t["title"])
st.write(t["subtitle"])

flash = st.session_state.pop("flash", None)  # message queued right before a rerun
if flash:
    getattr(st, flash[0])(flash[1])

# One shared password (APP_PASSWORD). This keeps strangers out; it is NOT per-user accounts.
saved_locations = storage.list_locations()
saved_names = dict(saved_locations)

# Multi-location login (Admin or specific restaurant client)
# ============================================================================
# Multi-Tenant Login & Sidebar Setup
# ============================================================================
for key, default in (
    ("auth_role", "admin"),
    ("locked_slug", None),
    ("force_login", False),
):
    if key not in st.session_state:
        st.session_state[key] = default

saved_locations = storage.list_locations()
saved_names = dict(saved_locations)

app_password = get_secret("APP_PASSWORD")
need_login = bool(app_password) or storage.has_any_location_passwords() or st.session_state.get("force_login", False)

if need_login and not st.session_state.get("authed", False):
    st.subheader("🔐 Login — TableCast Analytics")
    login_choices = ["👑 Admin (All Locations)"] + [f"🏪 {name} ({s})" for s, name in saved_locations]
    picked_login = st.selectbox("Select Account / Location", login_choices, key="login_account_select")
    entered = st.text_input(t["password_prompt"], type="password", key="password_input")
    
    if st.button("🔓 Sign In", key="login_submit_btn") or entered:
        if picked_login.startswith("👑"):
            expected_admin_pw = str(app_password) if app_password else "admin"
            if hmac.compare_digest(entered.encode("utf-8"), expected_admin_pw.encode("utf-8")):
                st.session_state.authed = True
                st.session_state.auth_role = "admin"
                st.session_state.locked_slug = None
                st.session_state.force_login = False
                st.session_state.pop("password_input", None)
                st.rerun()
            else:
                st.error(t["password_wrong"])
        else:
            target_slug = picked_login.rsplit("(", 1)[-1].rstrip(")")
            if storage.verify_location_password(target_slug, entered):
                st.session_state.authed = True
                st.session_state.auth_role = "client"
                st.session_state.locked_slug = target_slug
                st.session_state.force_login = False
                st.session_state.pop("password_input", None)
                st.rerun()
            else:
                st.error(t["password_wrong"])
    st.stop()
# ============================================================================
# Sidebar: location, upload, settings, events
# ============================================================================
pending_location = st.session_state.pop("pending_location", None)
if pending_location in saved_names:
    st.session_state["saved_location"] = pending_location

st.sidebar.header(t["location_header"])
if st.session_state.auth_role == "client" and st.session_state.locked_slug:
    # Client is locked strictly to their own restaurant
    chosen_slug = st.session_state.locked_slug
    loc_name = saved_names.get(chosen_slug, chosen_slug)
    slug = chosen_slug
    st.sidebar.success(f"🔒 {loc_name}")
    if st.sidebar.button("🚪 Log out", key="logout_btn"):
        st.session_state.authed = False
        st.rerun()
else:
    chosen_slug = st.sidebar.selectbox(
        t["location_saved"], [""] + list(saved_names),
        format_func=lambda s: t["location_none"] if not s else saved_names[s], key="saved_location",
    )
    loc_name = st.sidebar.text_input(
        t["location_name"], value=saved_names.get(chosen_slug, ""), key=f"loc_name_{chosen_slug or 'new'}"
    ).strip()
    slug = storage.slugify(loc_name) if loc_name else "default"

settings = storage.load_settings(slug) if slug in saved_names else {}

# Admin-only field to set or change this location's password
new_loc_password = ""
if st.session_state.auth_role == "admin":
    pw_status = "✅ Password set (type to change)" if settings.get("password_hash") else "🔑 Set Location Password (optional)"
    new_loc_password = st.sidebar.text_input(pw_status, type="password", key=f"loc_pw_{slug}").strip()

st.sidebar.header(t["upload_header"])
uploaded_file = st.sidebar.file_uploader(t["upload_prompt"], type=["csv"], key="csv_file_uploader")
st.sidebar.markdown(t["req_format"])

st.sidebar.markdown("---")
st.sidebar.subheader(t["settings_header"])
country_codes = list(COUNTRY_NAMES)
country = st.sidebar.selectbox(
    t["country_label"], country_codes,
    index=country_codes.index(settings["country"]) if settings.get("country") in country_codes else 0,
    format_func=lambda c: f"{COUNTRY_NAMES[c]} ({c})", key=f"country_{slug}",
)
station = st.sidebar.text_input(t["station_label"], value=settings.get("station", "Vienna"), key=f"station_{slug}").strip()
lunch_share = st.sidebar.slider(t["lunch_share"], 0, 100, int(settings.get("lunch_share", 40)), key=f"lunch_{slug}")
currency = COUNTRY_CURRENCIES.get(country, "€")
avg_spend = st.sidebar.number_input(
    t["avg_spend"].replace("€", currency),
    min_value=0.0, value=float(settings.get("avg_spend", 25.0)), step=1.0, key=f"spend_{slug}"
)
guests_per_server = st.sidebar.number_input(
    t["guests_per_server"], min_value=1, value=int(settings.get("guests_per_server", 15)), step=1, key=f"gps_{slug}"
)
recipients_text = st.sidebar.text_input(
    t["email_recipients"], value=settings.get("recipients", ""), key=f"recipients_{slug}"
)

st.sidebar.markdown("---")
st.sidebar.subheader(t["events_header"])
st.sidebar.write(t["events_desc"])
saved_events = storage.load_events(slug)
has_custom_event = st.sidebar.checkbox(t["checkbox_event"], key="private_event_checkbox")
event_date = None
event_extra_visitors = 0
if has_custom_event:
    event_date = st.sidebar.date_input(t["event_date"], value=date.today(), key="event_date_input")
    event_extra_visitors = int(
        st.sidebar.number_input(t["extra_guests"], min_value=1, max_value=500, value=25, key="extra_guests_input")
    )
    ev_col1, ev_col2 = st.sidebar.columns(2)
    if ev_col1.button("➕ Save Event", key="save_event_btn"):
        saved_events[event_date.isoformat()] = event_extra_visitors
        storage.save_events(slug, saved_events)
        st.rerun()
    if saved_events and ev_col2.button("🗑️ Clear All", key="clear_events_btn"):
        saved_events = storage.save_events(slug, {})
        st.rerun()

if saved_events:
    events_summary = ", ".join(f"{k} (+{v})" for k, v in sorted(saved_events.items()))
    st.sidebar.caption(f"📅 **Saved Events:** {events_summary}")

# ============================================================================
# Data: uploaded file, saved history, or both (new days are added to saved history)
# ============================================================================
saved_visits = storage.load_visits(slug) if slug in saved_names else None

if uploaded_file is None and saved_visits is None:
    st.session_state.ai_trained = False
    st.session_state.file_sig = None
    st.info(t["waiting_msg"])
    st.stop()

if uploaded_file is not None:
    try:
        raw_df = cached_read_csv(uploaded_file.getvalue())
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, csv.Error):
        st.error(t["err_read"])
        st.stop()
    # When history is already saved, a short upload of just the newest days is fine.
    upload_min_days = 1 if saved_visits is not None else core.MIN_DAYS
    upload_clean, notes, error = core.clean_visits(raw_df, min_days=upload_min_days)
    for level, key, kwargs in notes:
        getattr(st, level)(t[key].format(**kwargs))
    if error is not None:
        st.error(t[error[0]].format(**error[1]))
        st.stop()
    visits = storage.merge_visits(saved_visits, upload_clean)
    if len(visits) < core.MIN_DAYS:
        st.error(t["err_too_short"].format(min=core.MIN_DAYS, found=len(visits)))
        st.stop()
    st.success(t["success_loaded"].format(n=len(visits)))
    file_sig = (slug, uploaded_file.name, uploaded_file.size, len(visits))
else:
    visits = saved_visits
    st.info(t["loaded_saved"].format(n=len(visits)))
    file_sig = (slug, None, None, len(visits))

# New data (or another location) invalidates the previous "trained" state.
if st.session_state.file_sig != file_sig:
    st.session_state.file_sig = file_sig
    st.session_state.ai_trained = False

st.sidebar.markdown("---")
if st.sidebar.button(t["save_location_btn"], key="save_location_button"):
    if not loc_name:
        st.sidebar.error(t["err_location_name"])
    else:
        storage.save_visits(slug, visits)
        storage.save_settings(slug, {
            "name": loc_name, "language": selected_lang, "country": country, "station": station,
            "lunch_share": lunch_share, "avg_spend": avg_spend, "guests_per_server": int(guests_per_server),
            "recipients": recipients_text,
            "password_hash": storage.hash_password(new_loc_password) if new_loc_password else settings.get("password_hash", ""),
        })
        storage.sync_to_cloud(slug)
        st.session_state["pending_location"] = slug
        st.session_state["flash"] = ("success", t["saved_ok"])
        st.rerun()
        st.session_state["pending_location"] = slug
        st.session_state["flash"] = ("success", t["saved_ok"])
        st.rerun()

if st.sidebar.button(t["train_btn"], key="train_ai_button"):
    st.session_state.ai_trained = True
if not st.session_state.ai_trained:
    st.stop()

with st.spinner(t["spinner"]):
    bundle = cached_train(visits, station, country, saved_events)
if bundle is None:
    st.error(t["err_training"])
    st.stop()
if not bundle["weather_ok"]:
    st.warning(t["warn_weather"])

# ============================================================================
# Metrics and charts
# ============================================================================
summary_lines, baseline_text = reports.metric_lines(bundle, t)

col1, col2, col3, col4 = st.columns(4)
col1.metric(t["metric_visitors"], f"{bundle['mean_visitors']:.0f}")
col2.metric(t["metric_mae"], f"± {bundle['mae']:.2f} {t['unit_people']}")
col3.metric(t["metric_wape"], core.fmt_pct(bundle["wape"]))
col4.metric(t["metric_wape7"], core.fmt_pct(bundle["wape7"]))

st.caption(baseline_text)
if np.isfinite(bundle["wape7"]) and np.isfinite(bundle["base_wape7"]):
    if bundle["wape7"] < bundle["base_wape7"]:
        st.success(t["verdict_better"])
    else:
        st.warning(t["verdict_worse"])

st.subheader(t["main_chart_header"])
st.line_chart(
    bundle["chart"].rename(
        columns={"actual": t["legend_actual"], "pred": t["legend_pred"], "baseline": t["legend_baseline"]}
    )
)

st.markdown("---")
st.subheader(t["dow_header"])
st.write(t["dow_sub"])
dow_df = pd.DataFrame({"day": DAYS, "visitors": bundle["dow_means"].to_numpy()})
dow_chart = (
    alt.Chart(dow_df)
    .mark_bar()
    .encode(
        x=alt.X("day:N", sort=DAYS, title=None),
        y=alt.Y("visitors:Q", title=t["metric_visitors"]),
        tooltip=["day", alt.Tooltip("visitors:Q", format=".1f")],
    )
    .properties(width="container", height=320)
)
st.altair_chart(dow_chart)

# ============================================================================
# 7-day planner (with forecast ranges)
# ============================================================================
st.markdown("---")
st.subheader(t["planner_header"])

history = bundle["history"]
last_date = history.index[-1]
st.caption(t["forecast_from"].format(date=last_date.strftime("%Y-%m-%d")))
if bundle["closed_weekdays"]:
    st.caption(t["closed_caption"].format(days=", ".join(DAYS[d] for d in sorted(bundle["closed_weekdays"]))))

future_dates = core.future_dates_after(history)
auto_weather, live_weather_ok = core.fetch_future_weather(station, country, future_dates, history)

manual_override = st.checkbox(
    t.get("manual_weather_override", "Override automatic weather forecast manually (What-If scenario)"),
    value=not live_weather_ok,
    key="manual_weather_checkbox",
)
if manual_override:
    default_temp, _ = core.default_weather(history)
    col_b1, col_b2 = st.columns(2)
    pred_temp = col_b1.slider(t["avg_temp"], -10.0, 40.0, default_temp, key="temp_slider_input")
    pred_rain = col_b2.slider(t["avg_rain"], 0.0, 50.0, 0.0, key="rain_slider_input")
    future_weather = pd.DataFrame({"temp": pred_temp, "rain": pred_rain}, index=future_dates)
else:
    future_weather = auto_weather
    weather_summary = " | ".join(
        f"{DAYS[d.dayofweek][:3]}: {row['temp']:.0f}°C, {row['rain']:.1f}mm"
        for d, row in future_weather.iterrows()
    )
    st.caption(f"🌦️ **Open-Meteo 7-Day Forecast ({station}):** {weather_summary}")

base_preds = bundle["forecaster"].forecast(history.tail(core.LOOKBACK_DAYS), future_weather)

event_active = bool(has_custom_event and event_date is not None)
if event_active and event_date not in {d.date() for d in future_dates}:
    st.warning(t["warn_event_outside"])
    event_active = False

plan = core.build_plan(
    base_preds, bundle["intervals"], last_date,
    event_date if event_active else None, event_extra_visitors,
    lunch_share, avg_spend, int(guests_per_server),
    events=saved_events,
)
show_range = bundle["intervals"] is not None
headers, table_rows, pdf_rows = reports.format_plan(plan, t, DAYS, MONTHS, show_range, currency=currency)
forecast_df = pd.DataFrame(table_rows)
st.dataframe(forecast_df, hide_index=True)
if show_range:
    st.caption(t["range_caption"])

# ============================================================================
# Exports and email
# ============================================================================
pdf_bytes = reports.build_pdf(t["pdf_title"] + (f" - {loc_name}" if loc_name else ""), summary_lines, headers, pdf_rows)

d_col1, d_col2 = st.columns(2)
with d_col1:
    st.download_button(
        label=t["download_btn"],
        data=forecast_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="weekly_forecast_schedule.csv",
        mime="text/csv",
        key="download_csv_btn",
    )
with d_col2:
    st.download_button(
        label=t["download_pdf_btn"],
        data=pdf_bytes,
        file_name="executive_forecast_report.pdf",
        mime="application/pdf",
        key="download_pdf_btn",
    )

st.markdown("---")
st.subheader(t["email_header"])
st.caption(f"{t['email_recipients']}: {recipients_text or '-'}")
if st.button(t["email_send_btn"], key="send_email_button"):
    smtp = mailer.smtp_config(get_secret)
    recipients = mailer.parse_recipients(recipients_text)
    if smtp is None:
        st.error(t["email_not_configured"])
    elif not recipients:
        st.error(t["email_no_recipients"])
    else:
        message = mailer.build_email(
            smtp["from"], recipients,
            t["email_subject"].format(name=loc_name or "-", date=future_dates[0].strftime("%Y-%m-%d")),
            t["email_body"].format(
                name=loc_name or "-", date=last_date.strftime("%Y-%m-%d"),
                total=sum(row["expected"] for row in plan),
            ),
            pdf_bytes, "executive_forecast_report.pdf",
        )
        try:
            mailer.send_email(message, smtp)
            st.success(t["email_sent"].format(n=len(recipients)))
        except OSError as exc:  # includes smtplib.SMTPException
            st.error(t["email_failed"].format(reason=f"{type(exc).__name__}: {str(exc)[:200]}"))
st.caption(t["auto_email_hint"])

# ============================================================================
# Track record: saved forecasts vs what really happened
# ============================================================================
st.markdown("---")
st.subheader(t["tracking_header"])
st.write(t["tracking_desc"])

compared = storage.compare_log_to_actuals(storage.load_log(slug), visits)
if compared.empty:
    st.info(t["tracking_empty"])
else:
    record = storage.track_record_metrics(compared)
    st.write(t["tracking_metrics"].format(
        n=record["n"], wape=core.fmt_pct(record["wape"]), cov=core.fmt_pct(record["coverage"]),
    ))
    st.line_chart(
        compared.set_index("target_date")[["actual", "predicted"]].rename(
            columns={"actual": t["legend_actual"], "predicted": t["legend_pred"]}
        )
    )
if st.button(t["save_forecast_btn"], key="save_forecast_button"):
    saved_days = storage.log_forecast(slug, date.today(), plan)
    st.success(t["forecast_saved"].format(n=saved_days))

# ============================================================================
# Where saved data lives
# ============================================================================
st.markdown("---")
st.caption(t["storage_note"].format(path=storage.data_dir().resolve()))
