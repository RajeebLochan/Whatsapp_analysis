"""WhatsApp chat export parser.

Replaces the former ``preprcosseing.py``. It parses whole records line by line,
so a message and its timestamp always come from the same header line and the
first message is never discarded.

Supported header styles (examples use synthetic names):

Android-style ``<date>, <time> - <rest>``::

    1/5/24, 9:01 PM - Alice: hi
    05/01/2024, 21:01 - Alice: hi
    05.01.24, 21:01 - Alice: hi
    2024-01-05, 21:01 - Alice: hi

iOS-style ``[<date>, <time>] <rest>`` (optional seconds, optional leading
U+200E marks)::

    [05/01/2024, 21:01:07] Alice: hi
    [1/5/24, 9:01:07 PM] Alice: hi

Dates may use ``/``, ``.`` or ``-``; years may have 2 or 4 digits; times may be
12-hour (AM/PM, including ``a.m.``) or 24-hour with optional seconds. Narrow
and ordinary no-break spaces, BOMs and bidirectional marks around the header
are normalised. Message content is preserved exactly.

Timestamps are returned as *naive* datetimes. A WhatsApp export records no
timezone: times are in the exporting phone's local time and this module does
not invent one.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import pandas as pd

from . import config
from . import text as textlib

DateOrder = Literal["auto", "DMY", "MDY", "YMD"]
DATE_ORDER_LABELS = {
    "auto": "Detect automatically",
    "DMY": "Day / Month / Year",
    "MDY": "Month / Day / Year",
    "YMD": "Year / Month / Day",
}

# Bidirectional/format marks that WhatsApp (mostly iOS) places around headers.
_DIR_MARKS = "﻿‎‏‪‫‬‭‮⁦⁧⁨⁩"
_DIR_CLASS = f"[{_DIR_MARKS}]"
_SP = r"[ \t    ]"  # spaces seen between header fields
_DATE = r"(?P<d1>\d{1,4})(?P<sep>[/.\-])(?P<d2>\d{1,2})(?P=sep)(?P<d3>\d{2,4})\.?"
_TIME = (
    r"(?P<h>\d{1,2})[:.](?P<mi>\d{2})(?:[:.](?P<s>\d{2}))?"
    rf"(?:{_SP}*{_DIR_CLASS}*(?P<ampm>[AaPp]\.?{_SP}?[Mm]\.?))?"
)
_ANDROID_RE = re.compile(rf"^{_DIR_CLASS}*{_DATE},?{_SP}+{_DIR_CLASS}*{_TIME}{_DIR_CLASS}*{_SP}*[-–]{_SP}(?P<rest>.*)$")
_IOS_RE = re.compile(
    rf"^{_DIR_CLASS}*\[{_DATE},?{_SP}+{_DIR_CLASS}*{_TIME}{_DIR_CLASS}*\]{_SP}*{_DIR_CLASS}*(?P<rest>.*)$"
)
_LINE_SPLIT_RE = re.compile(r"\r\n|\r|\n")  # NOT str.splitlines(): keeps U+2028 etc. inside messages

# System events (English exports). Anchored to the start of the record and
# never allowed to cross a ':' so "Bob: I added you" is not a system event.
_SYSTEM_PHRASES = (
    r"added",
    r"removed",
    r"left",
    r"joined using this group's invite link",
    r"joined",
    r"created group",
    r"created this group",
    r"changed the subject",
    r"changed this group's",
    r"changed the group",
    r"changed their phone number",
    r"changed to \+?\d",
    r"deleted this group's",
    r"was added",
    r"was removed",
    r"pinned a message",
    r"turned on disappearing messages",
    r"turned off disappearing messages",
    r"changed the settings",
    r"reset this group's invite link",
    r"started a call",
    r"is now an admin",
    r"are now an admin",
    r"now an admin",
    r"requested to join",
    r"blocked this contact",
    r"unblocked this contact",
)
_SYSTEM_RE = re.compile(
    r"^(?:"
    r"messages and calls are end-to-end encrypted"
    r"|messages to this (?:group|chat) are now secured"
    r"|your security code with"
    r"|you're now an admin"
    r"|this chat is with a business account"
    r"|disappearing messages"
    r"|missed (?:voice|video|group voice|group video) call"
    r"|(?:voice|video) call\b"
    r"|[^:\n]{1,100}? (?:" + "|".join(_SYSTEM_PHRASES) + r")\b"
    r")",
    re.IGNORECASE,
)
_MEDIA_RE = re.compile(
    r"^(?:<media omitted>|<attached: [^>]*>|null"
    r"|(?:image|video|audio|sticker|gif|document|contact card|media|video note|voice message) omitted"
    r"|.+ \(file attached\)|.+ document omitted|view once (?:photo|video|voice message))$",
    re.IGNORECASE | re.DOTALL,
)
_DELETED_RE = re.compile(r"^(?:this message was deleted|you deleted this message)(?: by .+)?\.?$", re.IGNORECASE)
_EDITED_RE = re.compile(r"\s*<this message was edited>\s*$", re.IGNORECASE)
_LOCATION_RE = re.compile(r"^(?:live )?location: ", re.IGNORECASE)

MESSAGE_TYPES = ("text", "media", "deleted", "system")
COLUMNS = ["timestamp", "sender", "message", "text", "message_type", "edited", "line_no"]
FEATURE_COLUMNS = ["n_words", "n_links", "n_emoji"]  # derived once per upload (0 for non-text rows)


class ParseError(ValueError):
    """Raised with a user-safe message (never contains chat content)."""


@dataclass
class ParseIssue:
    line_no: int
    reason: str


@dataclass
class ParseReport:
    total_lines: int = 0
    records_found: int = 0
    messages_parsed: int = 0
    skipped_records: int = 0
    skipped_details: list[ParseIssue] = field(default_factory=list)
    preamble_lines: int = 0
    out_of_order: int = 0
    truncated: int = 0
    style: str = ""
    date_order: str = ""
    date_order_source: str = ""
    date_order_ambiguous: bool = False
    clock: str = ""
    has_seconds: bool = False
    year_digits: str = ""
    type_counts: dict[str, int] = field(default_factory=dict)
    participants: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def detected_format(self) -> str:
        parts = [self.style, f"date order {self.date_order}", self.clock]
        if self.has_seconds:
            parts.append("with seconds")
        if self.year_digits:
            parts.append(f"{self.year_digits}-digit year")
        return ", ".join(p for p in parts if p)

    def add_skip(self, line_no: int, reason: str) -> None:
        self.skipped_records += 1
        if len(self.skipped_details) < config.MAX_WARNINGS_KEPT:
            self.skipped_details.append(ParseIssue(line_no, reason))


@dataclass
class ParseResult:
    df: pd.DataFrame
    report: ParseReport


@dataclass
class _RawRecord:
    line_no: int
    style: str  # "android" | "ios"
    d1: str
    d2: str
    d3: str
    h: str
    mi: str
    s: str | None
    ampm: str | None
    rest: str
    extra: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Upload decoding
# ---------------------------------------------------------------------------
def read_upload(data: bytes, filename: str = "") -> tuple[str, list[str]]:
    """Decode an uploaded ``.txt`` or ``.zip`` export into text.

    Returns ``(text, warnings)``. Raises :class:`ParseError` with a safe message
    for oversize, binary, encrypted or otherwise unusable files.
    """
    warnings: list[str] = []
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise ParseError(f"The file is larger than the {config.MAX_UPLOAD_MB} MB limit.")
    if not data:
        raise ParseError("The uploaded file is empty.")

    if data[:4] == b"PK\x03\x04" or filename.lower().endswith(".zip"):
        data = _extract_zip_text(data, warnings)

    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16"), warnings
        except UnicodeDecodeError as exc:
            raise ParseError("The file looks like UTF-16 text but could not be decoded.") from exc
    if b"\x00" in data[:4096]:
        raise ParseError("This does not look like a text chat export (binary content detected).")
    try:
        return data.decode("utf-8-sig"), warnings
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace").lstrip("﻿")
        bad = text.count("�")
        warnings.append(f"The file is not valid UTF-8; {bad} undecodable byte sequence(s) were replaced with '�'.")
        return text, warnings


def _extract_zip_text(data: bytes, warnings: list[str]) -> bytes:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ParseError("The ZIP file could not be opened.") from exc
    with zf:
        infos = zf.infolist()
        if len(infos) > config.MAX_ZIP_MEMBERS:
            raise ParseError("The ZIP file contains too many entries. Export the chat 'Without media'.")
        txt = [i for i in infos if i.filename.lower().endswith(".txt") and not i.is_dir()]
        if not txt:
            raise ParseError("The ZIP file does not contain a .txt chat export.")
        preferred = [i for i in txt if i.filename.rsplit("/", 1)[-1].lower() == "_chat.txt"]
        info = (preferred or txt)[0]
        if len(txt) > 1:
            warnings.append(f"The ZIP contains {len(txt)} text files; only one chat export was read.")
        if info.flag_bits & 0x1:
            raise ParseError("Encrypted ZIP files are not supported.")
        if info.file_size > config.MAX_UPLOAD_BYTES:
            raise ParseError(f"The chat inside the ZIP is larger than the {config.MAX_UPLOAD_MB} MB limit.")
        with zf.open(info) as fh:
            content = fh.read(config.MAX_UPLOAD_BYTES + 1)  # do not trust the declared size
        if len(content) > config.MAX_UPLOAD_BYTES:
            raise ParseError(f"The chat inside the ZIP is larger than the {config.MAX_UPLOAD_MB} MB limit.")
        return content


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _match_header(line: str) -> tuple[str, re.Match[str]] | None:
    stripped = line.lstrip(_DIR_MARKS)
    if not stripped:
        return None
    first = stripped[0]
    if first == "[":
        m = _IOS_RE.match(line)
        return ("ios", m) if m else None
    if first.isdigit():
        m = _ANDROID_RE.match(line)
        return ("android", m) if m else None
    return None


def _collect_records(lines: list[str], report: ParseReport) -> list[_RawRecord]:
    records: list[_RawRecord] = []
    current: _RawRecord | None = None
    for idx, line in enumerate(lines, start=1):
        hit = _match_header(line)
        if hit is None:
            if current is None:
                if line.strip(_DIR_MARKS + " \t"):
                    report.preamble_lines += 1
            else:
                current.extra.append(line)
            continue
        style, m = hit
        current = _RawRecord(
            line_no=idx,
            style=style,
            d1=m["d1"],
            d2=m["d2"],
            d3=m["d3"],
            h=m["h"],
            mi=m["mi"],
            s=m["s"],
            ampm=m["ampm"],
            rest=m["rest"],
        )
        records.append(current)
    return records


def _resolve_year(raw: str) -> int | None:
    if len(raw) == 2:
        return 2000 + int(raw)
    if len(raw) == 4:
        return int(raw)
    return None


def _ymd(rec: _RawRecord, order: str) -> tuple[int | None, int, int]:
    a, b, c = rec.d1, rec.d2, rec.d3
    if order == "YMD":
        return _resolve_year(a), int(b), int(c)
    if order == "DMY":
        return _resolve_year(c), int(b), int(a)
    return _resolve_year(c), int(a), int(b)  # MDY


def _time_parts(rec: _RawRecord) -> tuple[int, int, int] | None:
    hour, minute = int(rec.h), int(rec.mi)
    second = int(rec.s) if rec.s else 0
    if rec.ampm:
        if not 1 <= hour <= 12:
            return None
        is_pm = rec.ampm.strip().lower().startswith("p")
        hour = (hour % 12) + (12 if is_pm else 0)
    elif not 0 <= hour <= 23:
        return None
    if not (0 <= minute <= 59 and 0 <= second <= 59):
        return None
    return hour, minute, second


def _to_datetime(rec: _RawRecord, order: str) -> datetime | None:
    year, month, day = _ymd(rec, order)
    t = _time_parts(rec)
    if year is None or t is None:
        return None
    try:
        return datetime(year, month, day, *t)
    except ValueError:
        return None


def _decide_date_order(records: list[_RawRecord], requested: str) -> tuple[str, str, bool]:
    """Return ``(order, source, ambiguous)``."""
    if requested in ("DMY", "MDY", "YMD"):
        return requested, "chosen by you", False
    if all(len(r.d1) == 4 for r in records):
        return "YMD", "unambiguous (year first)", False
    first_gt12 = sum(1 for r in records if len(r.d1) <= 2 and int(r.d1) > 12)
    second_gt12 = sum(1 for r in records if len(r.d1) <= 2 and int(r.d2) > 12)
    if first_gt12 and not second_gt12:
        return "DMY", "unambiguous (a day value above 12 appears first)", False
    if second_gt12 and not first_gt12:
        return "MDY", "unambiguous (a day value above 12 appears second)", False
    if first_gt12 and second_gt12:
        order = "DMY" if first_gt12 >= second_gt12 else "MDY"
        return order, "conflicting evidence; chose the majority — please check", True

    # Every day and month value is <= 12: genuinely ambiguous. Prefer the order
    # under which the export is (more) chronological, then a locale hint.
    def inversions(order: str) -> int:
        prev = None
        count = 0
        for r in records:
            dt = _to_datetime(r, order)
            if dt is None:
                continue
            if prev is not None and dt < prev:
                count += 1
            prev = dt
        return count

    inv_dmy, inv_mdy = inversions("DMY"), inversions("MDY")
    if inv_dmy != inv_mdy:
        order = "DMY" if inv_dmy < inv_mdy else "MDY"
        return order, "inferred from message order — please confirm", True
    no_leading_zero = all(not r.d1.startswith("0") and not r.d2.startswith("0") for r in records)
    if no_leading_zero and all(len(r.d3) == 2 for r in records) and any(r.ampm for r in records):
        return "MDY", "guessed from a US-style format (M/D/YY with AM/PM) — please confirm", True
    return "DMY", "guessed (ambiguous dates) — please confirm", True


def _classify(style: str, rest: str) -> tuple[str | None, str, str]:
    """Return ``(sender, original_message, message_type)``."""
    sender: str | None = None
    message = rest
    if style == "android" and _SYSTEM_RE.match(rest.lstrip(_DIR_MARKS)):
        return None, rest, "system"
    if ": " in rest:
        candidate, body = rest.split(": ", 1)
        candidate_clean = candidate.strip(_DIR_MARKS + " ")
        if 0 < len(candidate_clean) <= 100:
            sender, message = candidate_clean, body
    if sender is None:
        return None, rest, "system"

    probe = message.strip()
    starts_with_mark = bool(probe) and probe[0] in _DIR_MARKS
    probe = _EDITED_RE.sub("", probe.strip(_DIR_MARKS + " "))
    if _DELETED_RE.match(probe):
        return sender, message, "deleted"
    if _MEDIA_RE.match(probe):
        return sender, message, "media"
    if starts_with_mark and not _LOCATION_RE.match(probe):
        # iOS marks system notices inside a sender line with a leading U+200E.
        return None, message, "system"
    if style == "ios" and _SYSTEM_RE.match(probe) and probe.lower().startswith("messages and calls"):
        return None, message, "system"
    return sender, message, "text"


def derive_text(message: str) -> str:
    """Analysis text: export marks and the 'edited' marker removed, content kept."""
    text = _EDITED_RE.sub("", message)
    return text.strip(_DIR_MARKS + " \t\n")


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with word/link/emoji counts for text rows (see ``wa_analyzer.text``)."""
    texts = df["text"].where(df["message_type"] == "text", "").astype(str)
    return df.assign(
        n_words=texts.map(textlib.count_words).astype("int64"),
        n_links=texts.map(textlib.count_links).astype("int64"),
        n_emoji=texts.map(lambda t: len(textlib.extract_emoji(t))).astype("int64"),
    )


def parse_chat(text: str, date_order: DateOrder = "auto") -> ParseResult:
    """Parse an exported chat. Raises :class:`ParseError` for unusable input."""
    if date_order not in DATE_ORDER_LABELS:
        raise ValueError(f"unknown date_order {date_order!r}")
    report = ParseReport()
    if not text or not text.strip():
        raise ParseError("The chat export is empty.")
    lines = _LINE_SPLIT_RE.split(text)
    if lines and lines[-1] == "":
        lines.pop()
    report.total_lines = len(lines)
    if report.total_lines > config.MAX_LINE_COUNT:
        raise ParseError(
            f"The export has more than {config.MAX_LINE_COUNT:,} lines, which exceeds the processing limit."
        )

    records = _collect_records(lines, report)
    report.records_found = len(records)
    if not records:
        raise ParseError(
            "No WhatsApp messages were recognised. Supported formats are Android-style "
            "'date, time - Name: message' and iOS-style '[date, time] Name: message' exports."
        )
    if report.preamble_lines:
        report.warnings.append(f"{report.preamble_lines} line(s) before the first timestamp were ignored.")

    styles = Counter(r.style for r in records)
    style = styles.most_common(1)[0][0]
    report.style = (
        "iOS style ([date, time] Name: text)" if style == "ios" else "Android style (date, time - Name: text)"
    )
    if len(styles) > 1:
        report.warnings.append("The file mixes Android- and iOS-style lines.")
    has_ampm = sum(1 for r in records if r.ampm)
    report.clock = "12-hour" if has_ampm == len(records) else ("24-hour" if has_ampm == 0 else "mixed 12/24-hour")
    report.has_seconds = any(r.s for r in records)
    year_lengths = {len(r.d1) if len(r.d1) == 4 else len(r.d3) for r in records}
    report.year_digits = "/".join(str(n) for n in sorted(year_lengths))

    order, source, ambiguous = _decide_date_order(records, date_order)
    report.date_order, report.date_order_source, report.date_order_ambiguous = order, source, ambiguous

    rows: list[tuple] = []
    for rec in records:
        ts = _to_datetime(rec, order)
        if ts is None:
            report.add_skip(rec.line_no, "invalid date or time for the selected date order")
            continue
        while rec.extra and not rec.extra[-1].strip():
            rec.extra.pop()
        sender, message, mtype = _classify(rec.style, rec.rest)
        if rec.extra:
            message = message + "\n" + "\n".join(rec.extra)
        edited = bool(_EDITED_RE.search(message))
        rows.append((ts, sender, message, derive_text(message), mtype, edited, rec.line_no))

    if not rows:
        raise ParseError(
            "Timestamps were found but none could be read with the selected date order. Try a different date order."
        )
    if report.skipped_records:
        report.warnings.append(f"{report.skipped_records} record(s) were skipped because their timestamp was invalid.")
    if len(rows) > config.MAX_MESSAGES:
        report.truncated = len(rows) - config.MAX_MESSAGES
        rows = rows[-config.MAX_MESSAGES :]
        report.warnings.append(
            f"Only the most recent {config.MAX_MESSAGES:,} messages were kept "
            f"({report.truncated:,} older ones were dropped to stay within processing limits)."
        )

    df = pd.DataFrame.from_records(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    ts_values = df["timestamp"].to_numpy()
    report.out_of_order = int((ts_values[1:] < ts_values[:-1]).sum()) if len(df) > 1 else 0
    if report.out_of_order:
        report.warnings.append(
            f"{report.out_of_order} message(s) were out of chronological order in the file and were re-sorted "
            "(ties keep their original order)."
        )
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df = add_text_features(df)

    report.messages_parsed = len(df)
    report.type_counts = {t: int((df["message_type"] == t).sum()) for t in MESSAGE_TYPES}
    report.participants = int(df.loc[df["message_type"] != "system", "sender"].nunique())
    report.first_timestamp = df["timestamp"].iloc[0].to_pydatetime()
    report.last_timestamp = df["timestamp"].iloc[-1].to_pydatetime()
    if report.type_counts["system"] == len(df):
        report.warnings.append("The export contains only system notices; there are no participant messages.")
    elif report.type_counts["text"] == 0:
        report.warnings.append("The export contains no text messages (only media, deleted or system entries).")
    if ambiguous:
        report.warnings.append(
            "Day/month order is ambiguous in this export. Check the date range and change the date order if needed."
        )
    return ParseResult(df=df, report=report)
