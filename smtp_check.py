"""Step-by-step email connection test. Sends NOTHING.

    python smtp_check.py

Run it from the project folder. It reads .streamlit/secrets.toml (or environment variables),
tries your configured port, then the other common one (587 STARTTLS <-> 465 SSL), and says
which step fails and what that usually means.
"""
import smtplib
import socket
import ssl
import sys

import mailer


def attempt(host, port, user, password, implicit_ssl):
    """Walk through the connection. Returns (success, failed_step, exception)."""
    context = ssl.create_default_context()
    mode = "SSL (implicit)" if implicit_ssl else "STARTTLS"
    print(f"\n--- port {port}, {mode} ---")

    server = {}

    def dns():
        socket.getaddrinfo(host, port)

    def tcp():
        socket.create_connection((host, port), timeout=10).close()

    def handshake():
        if implicit_ssl:
            server["s"] = smtplib.SMTP_SSL(host, port, context=context, timeout=15)
            server["s"].ehlo()
        else:
            server["s"] = smtplib.SMTP(host, port, timeout=15)
            server["s"].ehlo()
            server["s"].starttls(context=context)
            server["s"].ehlo()

    def login():
        server["s"].login(user, password)

    steps = [
        ("DNS lookup of the server name", "dns", dns),
        ("TCP connection (is the port reachable?)", "tcp", tcp),
        (f"{mode} handshake", "handshake", handshake),
        ("login with your username + app password", "login", login),
    ]
    try:
        for label, name, fn in steps:
            try:
                fn()
                print(f"  OK    {label}")
            except Exception as exc:  # we want to report any failure, whatever its type
                print(f"  FAIL  {label}")
                print(f"        {type(exc).__name__}: {exc}")
                return False, name, exc
        return True, None, None
    finally:
        if "s" in server:
            try:
                server["s"].quit()
            except (smtplib.SMTPException, OSError):
                pass


def explain(step, exc):
    if step == "dns":
        return "The server name could not be found. Check SMTP_HOST (Gmail: smtp.gmail.com) and your internet connection."
    if step == "tcp":
        return ("Could not reach that port. A firewall, VPN, school/work network or your internet provider is probably "
                "blocking outgoing email. Try another network (e.g. your phone's hotspot).")
    if step == "handshake":
        if isinstance(exc, smtplib.SMTPServerDisconnected):
            return ("The connection was opened and then closed during the handshake. Most common causes: antivirus "
                    "'mail shield' / email scanning (Avast, Kaspersky, ESET, ...), a firewall or proxy, or the wrong "
                    "port/mode. Try the other port below, pause the antivirus email scanning, or try another network.")
        if isinstance(exc, ssl.SSLError):
            return ("The secure (TLS) connection failed. An antivirus or proxy that inspects encrypted traffic is a "
                    "common cause. Try another network or pause HTTPS/email scanning.")
        return "The server did not complete the secure handshake. Check the port (587 = STARTTLS, 465 = SSL)."
    if step == "login":
        if isinstance(exc, smtplib.SMTPAuthenticationError):
            return ("Gmail rejected the login. Use an APP PASSWORD (not your normal password): turn on 2-Step "
                    "Verification, then create one under Google Account > Security > App passwords. Use the full "
                    "email address as SMTP_USER.")
        return "Login failed. Check SMTP_USER and SMTP_PASSWORD."
    return "Unknown failure."


def main():
    cfg = mailer.smtp_config(mailer.env_or_secrets_getter())
    if cfg is None:
        print("No email settings found. Create .streamlit/secrets.toml in this folder with SMTP_HOST, SMTP_USER and")
        print("SMTP_PASSWORD (see README), or set them as environment variables, then run this again.")
        return 2

    host, port, user, password = cfg["host"], cfg["port"], cfg["user"], cfg["password"]
    print(f"Server:   {host!r}  port {port}")
    print(f"User:     {user!r}")
    print(f"Password: {len(password)} characters (not shown)")
    if host != host.strip() or " " in host:
        print("WARNING: SMTP_HOST contains a space. Remove it.")
    if " " in password:
        print("WARNING: the password contains spaces. A Gmail app password is 16 letters; remove the spaces.")
    if "@" not in user:
        print("WARNING: SMTP_USER should be the full email address.")

    first_ssl = port == 465
    results = []
    ok, step, exc = attempt(host, port, user, password, first_ssl)
    results.append((port, first_ssl, ok, step, exc))

    if not ok and step != "login":
        # also try the other common mode, in case the first one is the problem
        alt_port, alt_ssl = (587, False) if first_ssl else (465, True)
        ok2, step2, exc2 = attempt(host, alt_port, user, password, alt_ssl)
        results.append((alt_port, alt_ssl, ok2, step2, exc2))

    print("\n=== Result ===")
    working = [r for r in results if r[2]]
    if working:
        port_ok = working[0][0]
        print(f"Everything works on port {port_ok}.")
        if port_ok != port:
            print(f'Set SMTP_PORT = "{port_ok}" in .streamlit/secrets.toml and restart the app.')
        else:
            print("Your settings are fine. Restart the app and try sending again.")
        return 0
    for port_tried, _, _, step_failed, exc_failed in results:
        print(f"Port {port_tried}: failed at '{step_failed}'. {explain(step_failed, exc_failed)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
