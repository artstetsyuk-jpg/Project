"""Saved locations (history + settings) and the forecast track record.

Everything is plain files under a data folder (default ./data, override with the
FORECASTER_DATA_DIR environment variable):

    data/locations/<slug>/visits.csv      cleaned history (visit_date, visitors)
    data/locations/<slug>/settings.json   name, language, country, station, prices, recipients
    data/forecast_log.csv                 every forecast you chose to save
"""
import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

LOG_COLUMNS = ["location", "made_on", "target_date", "predicted", "low", "high"]


def data_dir():
    return Path(os.environ.get("FORECASTER_DATA_DIR", "data"))


def slugify(name):
    """'Café Zürich #2' -> 'cafe-zurich-2' (safe folder name). Empty names become 'default'."""
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug or "default"


def _location_dir(slug):
    return data_dir() / "locations" / slug


# ----------------------------------------------------------------------------
# Saved locations
# ----------------------------------------------------------------------------
def load_settings(slug):
    path = _location_dir(slug) / "settings.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(slug, settings):
    folder = _location_dir(slug)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "settings.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")


def load_events(slug):
    """Return saved custom events for a location as {'YYYY-MM-DD': extra_guests_int}."""
    path = _location_dir(slug) / "events.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in raw.items() if int(v) > 0}
    except (OSError, ValueError, TypeError):
        return {}


def save_events(slug, events):
    """Persist custom events dictionary {'YYYY-MM-DD': extra_guests_int} for a location."""
    folder = _location_dir(slug)
    folder.mkdir(parents=True, exist_ok=True)
    clean = {str(k): int(v) for k, v in (events or {}).items() if int(v) > 0}
    (folder / "events.json").write_text(json.dumps(clean, indent=2, sort_keys=True), encoding="utf-8")
    return clean

def list_locations():
    """[(slug, display_name), ...] for every location that has saved history."""
    root = data_dir() / "locations"
    if not root.exists():
        return []
    found = []
    for folder in sorted(root.iterdir()):
        if folder.is_dir() and (folder / "visits.csv").exists():
            found.append((folder.name, load_settings(folder.name).get("name", folder.name)))
    return found


def load_visits(slug):
    path = _location_dir(slug) / "visits.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["visit_date"])
    df["visitors"] = pd.to_numeric(df["visitors"], errors="coerce")
    return df.dropna().sort_values("visit_date").reset_index(drop=True)


def merge_visits(existing, new):
    """Combine saved history with newly uploaded days; for the same date the new value wins."""
    if existing is None or len(existing) == 0:
        return new.sort_values("visit_date").reset_index(drop=True)
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates("visit_date", keep="last").sort_values("visit_date")
    return combined.reset_index(drop=True)


def save_visits(slug, visits):
    """Save history, merged with whatever is already stored for this location."""
    merged = merge_visits(load_visits(slug), visits)
    folder = _location_dir(slug)
    folder.mkdir(parents=True, exist_ok=True)
    merged.to_csv(folder / "visits.csv", index=False)
    return merged


# ----------------------------------------------------------------------------
# Forecast track record
# ----------------------------------------------------------------------------
def _log_path():
    return data_dir() / "forecast_log.csv"


def load_log(slug):
    path = _log_path()
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    log = pd.read_csv(path)
    return log[log["location"] == slug].reset_index(drop=True)


def log_forecast(slug, made_on, plan):
    """Save the open days of a plan (see core.build_plan). Saving the same forecast twice on the
    same day replaces it instead of duplicating it. Returns the number of days saved."""
    new = pd.DataFrame([
        {
            "location": slug,
            "made_on": made_on.isoformat(),
            "target_date": row["date"].date().isoformat(),
            "predicted": row["expected"],
            "low": row["low"],
            "high": row["high"],
        }
        for row in plan if not row["is_closed"]
    ], columns=LOG_COLUMNS)
    if new.empty:
        return 0
    n_saved = len(new)

    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        same = (
            (old["location"] == slug)
            & (old["made_on"] == made_on.isoformat())
            & old["target_date"].isin(new["target_date"])
        )
        new = pd.concat([old[~same], new], ignore_index=True) if (~same).any() else new
    new.to_csv(path, index=False)
    return n_saved


def compare_log_to_actuals(log, visits):
    """Match saved forecasts to what really happened.

    For each target date uses the most recent forecast made BEFORE that day, then joins the
    real visitor count. Returns columns: target_date, predicted, low, high, actual.
    """
    columns = ["target_date", "predicted", "low", "high", "actual"]
    if log is None or len(log) == 0:
        return pd.DataFrame(columns=columns)
    log = log.copy()
    log["target_date"] = pd.to_datetime(log["target_date"])
    log["made_on"] = pd.to_datetime(log["made_on"])
    log = log[log["made_on"] < log["target_date"]]
    latest = log.sort_values("made_on").drop_duplicates("target_date", keep="last")
    actuals = visits.rename(columns={"visit_date": "target_date", "visitors": "actual"})
    merged = latest.merge(actuals, on="target_date", how="inner").sort_values("target_date")
    return merged[columns].reset_index(drop=True)


def track_record_metrics(compared):
    """{'n': days compared, 'wape': %, 'coverage': % of actuals inside the saved range (or None)}"""
    if len(compared) == 0:
        return {"n": 0, "wape": float("nan"), "coverage": None}
    actual = compared["actual"].to_numpy(dtype=float)
    predicted = compared["predicted"].to_numpy(dtype=float)
    total = actual.sum()
    wape = float(np.abs(actual - predicted).sum() / total * 100) if total > 0 else float("nan")
    has_range = compared["low"].notna() & compared["high"].notna()
    coverage = None
    if has_range.any():
        sub = compared[has_range]
        inside = (sub["actual"] >= sub["low"]) & (sub["actual"] <= sub["high"])
        coverage = float(inside.mean() * 100)
    return {"n": int(len(compared)), "wape": wape, "coverage": coverage}


# ============================================================================
# ADDON: Per-Location Passwords & Optional Supabase Cloud Sync
# ============================================================================
import hashlib
import hmac
import urllib.parse
import urllib.request


def hash_password(password: str) -> str:
    salt = os.urandom(16).hex()
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return f"pbkdf2_sha256${salt}${dk.hex()}"


def save_location_password(slug: str, raw_password: str):
    """Write password_hash directly into the location's settings.json file."""
    slug = slugify(slug)
    p = data_dir() / "locations" / slug / "settings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    data["password_hash"] = hash_password(raw_password)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _get_stored_hash(slug: str) -> str:
    p = data_dir() / "locations" / slugify(slug) / "settings.json"
    if not p.exists():
        return ""
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("password_hash", "")
    except Exception:
        return ""


def verify_location_password(slug: str, password: str) -> bool:
    if not password:
        return False
    stored_hash = _get_stored_hash(slug)
    if not stored_hash:
        return False
    if stored_hash.startswith("pbkdf2_sha256$"):
        _, salt, expected_hex = stored_hash.split("$", 2)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
        return hmac.compare_digest(dk.hex(), expected_hex)
    return hmac.compare_digest(password.encode("utf-8"), str(stored_hash).encode("utf-8"))


def has_any_location_passwords() -> bool:
    return any(bool(_get_stored_hash(s)) for s, _ in list_locations())


def _cloud_creds():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not (url and key):
        try:
            import streamlit as st
            url = url or st.secrets.get("SUPABASE_URL")
            key = key or st.secrets.get("SUPABASE_KEY")
        except Exception:
            pass
    return (url.rstrip("/"), key.strip()) if (url and key) else None


def is_cloud_enabled() -> bool:
    return _cloud_creds() is not None


def sync_to_cloud(slug: str):
    """Push local location files to Supabase if SUPABASE_URL & SUPABASE_KEY are configured."""
    creds = _cloud_creds()
    if not creds:
        return
    base_url, api_key = creds
    slug = slugify(slug)
    loc_folder = data_dir() / "locations" / slug
    if not loc_folder.exists():
        return

    payload = {"slug": slug}
    for fname, col in (
        ("settings.json", "settings_json"),
        ("events.json", "events_json"),
        ("visits.csv", "visits_csv"),
        ("log.csv", "log_csv"),
        ("forecast_log.csv", "log_csv"),
    ):
        fpath = loc_folder / fname
        if fpath.exists():
            payload[col] = fpath.read_text(encoding="utf-8")

    if _log_path().exists():
        payload["log_csv"] = _log_path().read_text(encoding="utf-8")

    req = urllib.request.Request(
        f"{base_url}/rest/v1/tablecast_locations",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "apikey": api_key,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8):
            pass
    except Exception:
        pass


def pull_all_from_cloud():
    """Restore any cloud-saved locations into the local ./data/locations/ folder on startup."""
    creds = _cloud_creds()
    if not creds:
        return
    base_url, api_key = creds
    req = urllib.request.Request(
        f"{base_url}/rest/v1/tablecast_locations?select=*",
        headers={"apikey": api_key, "Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
        for row in rows or []:
            slug = slugify(row.get("slug", ""))
            if not slug:
                continue
            loc_folder = data_dir() / "locations" / slug
            loc_folder.mkdir(parents=True, exist_ok=True)
            for fname, col in (
                ("settings.json", "settings_json"),
                ("events.json", "events_json"),
                ("visits.csv", "visits_csv"),
                ("log.csv", "log_csv"),
            ):
                if row.get(col) and not (loc_folder / fname).exists():
                    (loc_folder / fname).write_text(row[col], encoding="utf-8")
    except Exception:
        pass
