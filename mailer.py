"""Email settings and sending. Standard library only (no Streamlit, pandas or PDF dependencies),
so smtp_check.py can use it on its own."""
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path


def env_or_secrets_getter(secrets_path=".streamlit/secrets.toml"):
    """Getter that checks environment variables first, then a Streamlit-style secrets.toml.
    Lets weekly_report.py use the same settings file as the app (needs Python 3.11+ for the file part)."""
    values = {}
    path = Path(secrets_path)
    if path.exists():
        try:
            import tomllib
            values = tomllib.loads(path.read_text(encoding="utf-8"))
        except (ImportError, OSError, ValueError):
            values = {}

    def getter(name):
        return os.environ.get(name) or (str(values[name]) if name in values else None)
    return getter


def smtp_config(getter=os.environ.get):
    """Read SMTP settings through `getter(name) -> str | None` (environment variables by default).
    Returns None when the required settings are missing."""
    host, user, password = getter("SMTP_HOST"), getter("SMTP_USER"), getter("SMTP_PASSWORD")
    if not (host and user and password):
        return None
    try:
        port = int(getter("SMTP_PORT") or 587)
    except ValueError:
        port = 587
    return {"host": host, "port": port, "user": user, "password": password, "from": getter("SMTP_FROM") or user}


def parse_recipients(text):
    """'a@x.com, b@y.org; c@z.net' -> ['a@x.com', 'b@y.org', 'c@z.net'] (invalid entries dropped)."""
    parts = [p.strip() for chunk in str(text or "").split(";") for p in chunk.split(",")]
    return [p for p in parts if "@" in p and "." in p.split("@")[-1] and " " not in p]


def build_email(sender, recipients, subject, body, attachment_bytes=None, attachment_name="report.pdf"):
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(body)
    if attachment_bytes is not None:
        msg.add_attachment(attachment_bytes, maintype="application", subtype="pdf", filename=attachment_name)
    return msg


def send_email(msg, cfg):
    """Send via SMTP (STARTTLS on 587 by default, implicit SSL on 465). Raises OSError
    (which includes smtplib.SMTPException) on failure."""
    context = ssl.create_default_context()
    if cfg["port"] == 465:
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=context, timeout=30) as server:
            server.login(cfg["user"], cfg["password"])
            server.send_message(msg)
    else:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=30) as server:
            server.starttls(context=context)
            server.login(cfg["user"], cfg["password"])
            server.send_message(msg)
