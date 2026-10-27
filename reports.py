"""Plan formatting and the PDF report (no Streamlit dependency). Email lives in mailer.py."""
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from core import fmt_pct

PDF_SHARES = {
    6: [0.20, 0.12, 0.12, 0.20, 0.14, 0.22],   # date, expected, range, shift, revenue, servers
    5: [0.26, 0.14, 0.22, 0.15, 0.23],         # same without the range column
}


# ----------------------------------------------------------------------------
# Plan -> table rows
# ----------------------------------------------------------------------------
def format_plan(plan, t, days, months, show_range, currency="€"):
    """Returns (headers, table_rows, pdf_rows).

    table_rows: list of dicts (numbers stay numbers) for the on-screen table / CSV.
    pdf_rows:   list of preformatted string lists for the PDF.
    """
    rev_header = f"{t['col_revenue']} ({currency})" if currency else t["col_revenue"]
    headers = [t["col_date"], t["col_traffic"]]
    if show_range:
        headers.append(t["col_range"])
    headers += [t["col_shift"], rev_header, t["col_servers"]]

    table_rows, pdf_rows = [], []
    for row in plan:
        d = row["date"]
        label = f"{days[d.dayofweek]}, {months[d.month - 1]} {d.day:02d}"
        if row["on_event"]:
            label += f" 🎉 [{t['event_tag']}]"
        if row["is_closed"]:
            shift_text = t["closed"]
            servers_text = t["closed"]
        else:
            shift_text = f"{t['shift_lunch']}: {row['lunch']} | {t['shift_dinner']}: {row['dinner']}"
            servers_text = (
                f"{t['shift_lunch']}: {row['lunch_servers']} | "
                f"{t['shift_dinner']}: {row['dinner_servers']}"
            )
        range_text = "-" if row["is_closed"] or row["low"] is None else f"{row['low']} - {row['high']}"

        record = {t["col_date"]: label, t["col_traffic"]: row["expected"]}
        pdf_row = [label, str(row["expected"])]
        if show_range:
            record[t["col_range"]] = range_text
            pdf_row.append(range_text)
        record[t["col_shift"]] = shift_text
        record[rev_header] = row["revenue"]
        record[t["col_servers"]] = servers_text
        pdf_row += [shift_text, f"{row['revenue']:,.2f} {currency}".strip(), servers_text]
        table_rows.append(record)
        pdf_rows.append(pdf_row)
    return headers, table_rows, pdf_rows


def metric_lines(bundle, t):
    """The summary lines printed on the PDF. Returns (lines, baseline_text)."""
    baseline_text = t["baseline_caption"].format(
        b1=fmt_pct(bundle["base_wape"]), b7=fmt_pct(bundle["base_wape7"])
    )
    lines = [
        f"{t['metric_visitors']}: {bundle['mean_visitors']:.0f}  |  "
        f"{t['metric_mae']}: {bundle['mae']:.1f} {t['unit_people']}",
        f"{t['metric_wape']}: {fmt_pct(bundle['wape'])}  |  {t['metric_wape7']}: {fmt_pct(bundle['wape7'])}",
        baseline_text,
    ]
    return lines, baseline_text


# ----------------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------------
PDF_CHAR_MAP = {
    "€": "EUR",
    "œ": "oe",
    "Œ": "OE",
    "—": "-",
    "–": "-",
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "…": "...",
}


def _latin1(text):
    """Built-in PDF fonts only support Latin-1: transliterate ligatures/symbols before encoding."""
    s = str(text)
    for src, dst in PDF_CHAR_MAP.items():
        s = s.replace(src, dst)
    cleaned = s.encode("latin-1", "ignore").decode("latin-1")
    return " ".join(cleaned.split())


def build_pdf(title, summary_lines, headers, rows):
    """Returns the PDF as bytes (nothing is written to disk)."""
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _latin1(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(240, 240, 240)
    for line in summary_lines:
        pdf.cell(0, 8, _latin1(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C", fill=True)
    pdf.ln(8)

    shares = PDF_SHARES.get(len(headers)) or [1 / len(headers)] * len(headers)
    widths = [pdf.epw * share for share in shares]
    pdf.set_font("Helvetica", "B", 8)
    for width, header in zip(widths, headers):
        pdf.cell(width, 8, _latin1(header), border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    for row in rows:
        for width, value in zip(widths, row):
            pdf.cell(width, 8, _latin1(value), border=1, align="C")
        pdf.ln()
    return bytes(pdf.output())
