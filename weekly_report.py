"""Weekly report: train every saved location, build the PDF plan and email it.

Run it on a schedule (cron / Task Scheduler / GitHub Actions, see README.md):

    python weekly_report.py               # email every location that has recipients
    python weekly_report.py --dry-run     # build PDFs into data/out/, send nothing, log nothing
    python weekly_report.py --location cafe-zurich

Needs SMTP_HOST, SMTP_USER, SMTP_PASSWORD (and optionally SMTP_PORT, SMTP_FROM), either as environment
variables or in .streamlit/secrets.toml (the same file the app uses; run the script from the project folder).
Uses the same model and plan logic as the app. Weather is assumed neutral (last week's average
temperature, no rain), because there is nobody to move the weather inputs.
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

import core
import mailer
import reports
import storage
from i18n import DAY_TRANSLATIONS, MONTH_TRANSLATIONS, TRANSLATIONS

STALE_NOTE_AFTER_DAYS = 3   # add a warning to the email when the newest data is older than this
MAX_STALE_DAYS = 21         # refuse to forecast when the newest data is older than this


def build_location_report(slug, settings, today):
    """Returns (pdf_bytes, email_subject, email_body, plan) or raises ValueError with a reason."""
    lang = settings.get("language") if settings.get("language") in TRANSLATIONS else "English"
    t, days, months = TRANSLATIONS[lang], DAY_TRANSLATIONS[lang], MONTH_TRANSLATIONS[lang]
    name = settings.get("name", slug)

    visits = storage.load_visits(slug)
    if visits is None or len(visits) < core.MIN_DAYS:
        raise ValueError(f"needs at least {core.MIN_DAYS} saved days of history")
    last_date = visits["visit_date"].max()
    start = max(pd.Timestamp(today), last_date + pd.Timedelta(days=1))
    stale_days = (start - last_date).days - 1
    if stale_days > MAX_STALE_DAYS:
        raise ValueError(
            f"newest saved data is from {last_date.date()} ({stale_days} days old); upload recent data first"
        )
    loc_station = settings.get("station", "Vienna")
    loc_country = settings.get("country", "AT")
    saved_events = storage.load_events(slug)
    bundle = core.train_bundle(visits, loc_station, loc_country, events=saved_events)
    if bundle is None:
        raise ValueError("not enough usable history to train")
    history = bundle["history"]

    # Forecast from the day after the newest data through the end of the planned week. If the saved
    # data is stale this also covers the gap days (recursively), then we keep only the planned week.
    gap_to_end = pd.date_range(last_date + pd.Timedelta(days=1), start + pd.Timedelta(days=core.HORIZON - 1), freq="D")
    future_weather, _ = core.fetch_future_weather(loc_station, loc_country, gap_to_end, history)
    all_preds = bundle["forecaster"].forecast(history.tail(core.LOOKBACK_DAYS), future_weather)
    week_preds = all_preds.loc[start:start + pd.Timedelta(days=core.HORIZON - 1)]

    plan = core.build_plan(
        week_preds, bundle["intervals"], last_date, None, 0,
        int(settings.get("lunch_share", 40)),
        float(settings.get("avg_spend", 25.0)),
        int(settings.get("guests_per_server", 15)),
        events=saved_events,
    )
    headers, _, pdf_rows = reports.format_plan(plan, t, days, months, bundle["intervals"] is not None)
    lines, _ = reports.metric_lines(bundle, t)
    pdf_bytes = reports.build_pdf(f"{t['pdf_title']} - {name}", lines, headers, pdf_rows)

    body = t["email_body"].format(
        name=name, date=(start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        total=sum(row["expected"] for row in plan),
    )
    if stale_days > STALE_NOTE_AFTER_DAYS:
        body += "\n\n" + t["email_stale_note"].format(date=last_date.strftime("%Y-%m-%d"), n=stale_days)
    subject = t["email_subject"].format(name=name, date=start.strftime("%Y-%m-%d"))
    return pdf_bytes, subject, body, plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="write PDFs to data/out/ instead of emailing")
    parser.add_argument("--location", help="only this location slug (default: all with recipients)")
    args = parser.parse_args(argv)

    cfg = None if args.dry_run else mailer.smtp_config(mailer.env_or_secrets_getter())
    if not args.dry_run and cfg is None:
        print("SMTP is not configured (SMTP_HOST, SMTP_USER, SMTP_PASSWORD). Use --dry-run to test.")
        return 2

    today = date.today()
    failures = 0
    for slug, name in storage.list_locations():
        if args.location and slug != args.location:
            continue
        settings = storage.load_settings(slug)
        recipients = mailer.parse_recipients(settings.get("recipients", ""))
        if not recipients and not args.dry_run:
            print(f"[skip] {name}: no recipients saved")
            continue
        try:
            pdf_bytes, subject, body, plan = build_location_report(slug, settings, today)
            if args.dry_run:
                out_dir = storage.data_dir() / "out"
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"{slug}_{today.isoformat()}.pdf"
                path.write_bytes(pdf_bytes)
                print(f"[dry-run] {name}: wrote {path}")
            else:
                msg = mailer.build_email(cfg["from"], recipients, subject, body, pdf_bytes, f"forecast_{slug}.pdf")
                mailer.send_email(msg, cfg)
                storage.log_forecast(slug, today, plan)  # feeds the track record
                print(f"[sent] {name}: {len(recipients)} recipient(s)")
        except (ValueError, OSError) as exc:
            failures += 1
            print(f"[error] {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
