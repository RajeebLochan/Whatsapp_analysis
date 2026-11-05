"""Aggregate downloads: CSV tables and a short Markdown report.

Raw conversation text is never included. User-controlled text fields
(participant names, words, emoji) are protected against spreadsheet formula
injection following OWASP's CSV-injection guidance: a cell beginning with
``=``, ``+``, ``-``, ``@``, tab or carriage return (also after leading
whitespace) is prefixed with a single quote, and all fields are quoted.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Mapping
from datetime import datetime

import pandas as pd

from .analytics import Overview
from .parsing import ParseReport

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n", "＝", "＋", "－", "＠")


def csv_safe(value: object) -> object:
    """Neutralise a value that a spreadsheet could interpret as a formula."""
    if not isinstance(value, str) or not value:
        return value
    if value.startswith(_FORMULA_PREFIXES) or value.lstrip("  ").startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def to_safe_csv(df: pd.DataFrame) -> bytes:
    safe = df.copy()
    for col in safe.columns:
        if safe[col].dtype == object or pd.api.types.is_string_dtype(safe[col]):
            safe[col] = safe[col].map(csv_safe)
    safe.columns = [csv_safe(str(c)) for c in safe.columns]
    buf = io.StringIO()
    safe.to_csv(buf, index=False, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
    return buf.getvalue().encode("utf-8-sig")  # BOM so spreadsheet apps detect UTF-8


def pseudonym_map(names: list[str]) -> dict[str, str]:
    """Stable "Participant A/B/…" labels in the given order."""
    labels = {}
    for i, name in enumerate(names):
        letters = ""
        n = i
        while True:
            letters = chr(ord("A") + n % 26) + letters
            n = n // 26 - 1
            if n < 0:
                break
        labels[name] = f"Participant {letters}"
    return labels


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def overview_table(ov: Overview) -> pd.DataFrame:
    rows = [
        ("Messages (text + media + deleted)", ov.messages),
        ("Text messages", ov.text_messages),
        ("Media placeholders", ov.media_messages),
        ("Deleted-message markers", ov.deleted_messages),
        ("System events (not counted as messages)", ov.system_events),
        ("Words in text messages", ov.words),
        ("Links", ov.links),
        ("Emoji", ov.emoji),
        ("Participants with messages", ov.participants),
        ("First message", ov.first_message),
        ("Last message", ov.last_message),
        ("Calendar span (days, inclusive)", ov.calendar_days),
        ("Active days (≥1 message)", ov.active_days),
        ("Average messages per active day", ov.avg_per_active_day),
        ("Average messages per calendar day", ov.avg_per_calendar_day),
    ]
    return pd.DataFrame({"metric": [r[0] for r in rows], "value": [_fmt(r[1]) for r in rows]})


def build_report(
    report: ParseReport,
    ov: Overview,
    participant_table: pd.DataFrame,
    filter_description: str,
    generated_at: datetime | None = None,
) -> str:
    """Concise Markdown report without any message text."""
    generated_at = generated_at or datetime.now()
    lines = [
        "# WhatsApp chat summary report",
        "",
        f"Generated: {generated_at:%Y-%m-%d %H:%M} (times in the export's local time; no timezone recorded)",
        "",
        "## Parsing",
        f"- Detected format: {report.detected_format}",
        f"- Date order: {report.date_order} ({report.date_order_source})",
        f"- Records parsed: {report.messages_parsed:,}; skipped: {report.skipped_records:,}",
    ]
    lines += [f"- Warning: {w}" for w in report.warnings]
    lines += ["", "## Selection", f"- {filter_description}", "", "## Overview"]
    for _, row in overview_table(ov).iterrows():
        lines.append(f"- {row['metric']}: {row['value']}")
    lines += ["", "## Participants", "", "| Participant | Messages | Share % |", "|---|---:|---:|"]
    for _, row in participant_table.head(50).iterrows():
        name = str(row["participant"]).replace("|", "\\|")
        lines.append(f"| {name} | {int(row['messages']):,} | {row['share_pct']} |")
    lines += [
        "",
        "## Notes",
        "- Counts exclude system events. Media and deleted entries are placeholders in the export, not content.",
        "- This report contains aggregate statistics only; no message text is included.",
    ]
    return "\n".join(lines) + "\n"


def build_zip(tables: Mapping[str, pd.DataFrame], report_md: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, table in tables.items():
            zf.writestr(f"{name}.csv", to_safe_csv(table))
        zf.writestr("report.md", report_md.encode("utf-8"))
    return buf.getvalue()
