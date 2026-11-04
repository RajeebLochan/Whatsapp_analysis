"""Parser tests. All chats are synthetic."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime

import pandas as pd
import pytest

from wa_analyzer import config
from wa_analyzer.parsing import ParseError, parse_chat, read_upload

NNBSP = " "
LRM = "‎"


def _rows(result):
    df = result.df
    senders = [None if pd.isna(s) else s for s in df["sender"]]
    return list(zip(df["timestamp"].dt.to_pydatetime(), senders, df["text"], df["message_type"], strict=True))


# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------
def test_android_us_12h_preserves_first_message():
    chat = "1/5/24, 9:01 AM - Alice: first\n1/5/24, 9:02 AM - Bob: second\n1/13/24, 10:15 PM - Alice: third\n"
    r = parse_chat(chat)
    assert _rows(r) == [
        (datetime(2024, 1, 5, 9, 1), "Alice", "first", "text"),
        (datetime(2024, 1, 5, 9, 2), "Bob", "second", "text"),
        (datetime(2024, 1, 13, 22, 15), "Alice", "third", "text"),
    ]
    assert r.report.date_order == "MDY"
    assert r.report.clock == "12-hour"
    assert not r.report.date_order_ambiguous


def test_single_message_chat():
    r = parse_chat("1/5/24, 9:01 AM - Alice: only message\n")
    assert len(r.df) == 1
    assert r.df.loc[0, "text"] == "only message"


def test_android_narrow_nbsp_before_am_pm():
    chat = f"12/31/23, 11:59{NNBSP}PM - Alice: last of year\n1/1/24, 12:00{NNBSP}AM - Bob: midnight\n1/1/24, 12:30{NNBSP}PM - Bob: noon\n"
    r = parse_chat(chat)
    assert [t for t, *_ in _rows(r)] == [
        datetime(2023, 12, 31, 23, 59),
        datetime(2024, 1, 1, 0, 0),
        datetime(2024, 1, 1, 12, 30),
    ]


def test_dmy_24h_four_digit_year():
    chat = "05/01/2024, 09:00 - Alice: hi\n13/01/2024, 21:30 - Bob: hello\n"
    r = parse_chat(chat)
    assert [t for t, *_ in _rows(r)] == [datetime(2024, 1, 5, 9, 0), datetime(2024, 1, 13, 21, 30)]
    assert r.report.date_order == "DMY"
    assert r.report.clock == "24-hour"
    assert r.report.year_digits == "4"


def test_dotted_dates_and_iso_dates():
    r = parse_chat("13.02.24, 18:05 - Ana: hola\n")
    assert r.df.loc[0, "timestamp"] == datetime(2024, 2, 13, 18, 5)
    r = parse_chat("2024-02-13, 18:05 - Ana: hola\n")
    assert r.df.loc[0, "timestamp"] == datetime(2024, 2, 13, 18, 5)
    assert r.report.date_order == "YMD"


def test_ios_bracketed_with_seconds_and_marks():
    chat = (
        f"{LRM}[13/02/2024, 18:05:07] Ana: hola\n"
        f"[13/02/2024, 18:06:00] Ben: {LRM}image omitted\n"
        f"[13/02/2024, 18:07:30] Group: {LRM}Ana added Ben\n"
        f"[2/13/24, 6:08:09{NNBSP}PM] Ana: 12h variant\n"
    )
    # mixed date orders in one file are not realistic; parse the first three only
    r = parse_chat("".join(chat.splitlines(keepends=True)[:3]))
    assert _rows(r) == [
        (datetime(2024, 2, 13, 18, 5, 7), "Ana", "hola", "text"),
        (datetime(2024, 2, 13, 18, 6, 0), "Ben", "image omitted", "media"),
        (datetime(2024, 2, 13, 18, 7, 30), None, "Ana added Ben", "system"),
    ]
    assert r.report.style.startswith("iOS")
    assert r.report.has_seconds
    r2 = parse_chat(chat.splitlines(keepends=True)[3])
    assert r2.df.loc[0, "timestamp"] == datetime(2024, 2, 13, 18, 8, 9)


def test_lowercase_and_dotted_am_pm():
    r = parse_chat("13/02/2024, 6:08 p.m. - Ana: a\n13/02/2024, 6:09 am - Ana: b\n")
    assert sorted(r.df["timestamp"]) == [datetime(2024, 2, 13, 6, 9), datetime(2024, 2, 13, 18, 8)]


def test_bom_at_file_start_is_ignored():
    raw = "﻿05/01/2024, 09:00 - Alice: hi\n".encode()
    text, warnings = read_upload(raw, "chat.txt")
    r = parse_chat(text)
    assert r.df.loc[0, "sender"] == "Alice"
    assert warnings == []


# ---------------------------------------------------------------------------
# Content handling
# ---------------------------------------------------------------------------
def test_multiline_and_colons_in_content_and_alignment():
    chat = (
        "05/01/2024, 09:00 - Alice: line one\n"
        "line two: with colon\n"
        "\n"
        "line four\n"
        "05/01/2024, 09:05 - Bob: time is 10:30: ok?\n"
    )
    r = parse_chat(chat)
    assert r.df.loc[0, "message"] == "line one\nline two: with colon\n\nline four"
    assert r.df.loc[1, "sender"] == "Bob"
    assert r.df.loc[1, "text"] == "time is 10:30: ok?"
    assert r.df.loc[1, "timestamp"] == datetime(2024, 1, 5, 9, 5)
    assert list(r.df["line_no"]) == [1, 5]


def test_phone_number_and_unicode_senders_and_emoji_preserved():
    chat = (
        "05/01/2024, 09:00 - +91 98765 43210: नमस्ते 🙏🏾\n"
        "05/01/2024, 09:01 - Søren Ø: 👨‍👩‍👧 ok\n"
        "05/01/2024, 09:02 - 陈伟: 你好\n"
    )
    r = parse_chat(chat)
    assert list(r.df["sender"]) == ["+91 98765 43210", "Søren Ø", "陈伟"]
    assert r.df.loc[0, "text"] == "नमस्ते 🙏🏾"
    assert r.df.loc[1, "text"] == "👨‍👩‍👧 ok"


def test_unicode_line_separator_inside_message_is_not_a_new_line():
    r = parse_chat("05/01/2024, 09:00 - Alice: a b\n")
    assert r.df.loc[0, "text"] == "a b"


def test_system_media_deleted_edited_classification():
    chat = (
        "05/01/2024, 09:00 - Messages and calls are end-to-end encrypted. Learn more.\n"
        '05/01/2024, 09:01 - Alice changed the subject from "a: b" to "c"\n'
        "05/01/2024, 09:02 - Alice added Bob\n"
        "05/01/2024, 09:03 - Bob: I added you to the list\n"
        "05/01/2024, 09:04 - Bob: <Media omitted>\n"
        "05/01/2024, 09:05 - Bob: This message was deleted\n"
        "05/01/2024, 09:06 - Alice: You deleted this message\n"
        "05/01/2024, 09:07 - Alice: fixed typo <This message was edited>\n"
        "05/01/2024, 09:08 - Alice: IMG-001.jpg (file attached)\n"
        "05/01/2024, 09:09 - Bob left\n"
    )
    r = parse_chat(chat)
    assert list(r.df["message_type"]) == [
        "system",
        "system",
        "system",
        "text",
        "media",
        "deleted",
        "deleted",
        "text",
        "media",
        "system",
    ]
    assert r.df.loc[3, "sender"] == "Bob"
    assert r.df.loc[7, "text"] == "fixed typo"
    assert bool(r.df.loc[7, "edited"]) is True
    assert r.df.loc[7, "message"].endswith("<This message was edited>")  # original preserved
    assert r.report.type_counts == {"text": 2, "media": 2, "deleted": 2, "system": 4}
    assert r.report.participants == 2


# ---------------------------------------------------------------------------
# Date order
# ---------------------------------------------------------------------------
AMBIGUOUS = "01/02/2024, 09:00 - Alice: a\n03/04/2024, 09:00 - Bob: b\n"


def test_ambiguous_dates_flagged_and_explicit_choice_respected():
    auto = parse_chat(AMBIGUOUS)
    assert auto.report.date_order_ambiguous
    assert "please confirm" in auto.report.date_order_source
    dmy = parse_chat(AMBIGUOUS, date_order="DMY")
    mdy = parse_chat(AMBIGUOUS, date_order="MDY")
    assert dmy.df.loc[0, "timestamp"] == datetime(2024, 2, 1, 9, 0)
    assert mdy.df.loc[0, "timestamp"] == datetime(2024, 1, 2, 9, 0)
    assert not dmy.report.date_order_ambiguous
    assert dmy.report.date_order_source == "chosen by you"


def test_ambiguity_resolved_by_chronological_order():
    # Under DMY this goes 12 Jan -> 1 Feb; under MDY it goes Dec 1 -> Jan 2 (2024), out of order.
    chat = "12/01/2024, 09:00 - A: x\n01/02/2024, 09:00 - B: y\n"
    r = parse_chat(chat)
    assert r.report.date_order == "DMY"
    assert r.report.date_order_ambiguous
    assert "message order" in r.report.date_order_source


def test_invalid_dates_reported_not_invented():
    chat = "31/01/2024, 09:00 - A: ok\n31/02/2024, 09:00 - B: impossible date\n01/02/2024, 25:00 - C: bad hour\n"
    r = parse_chat(chat)
    assert list(r.df["sender"]) == ["A"]
    assert r.report.skipped_records == 2
    assert [d.line_no for d in r.report.skipped_details] == [2, 3]
    assert any("skipped" in w for w in r.report.warnings)


def test_explicit_wrong_order_skips_impossible_dates():
    r = parse_chat("13/01/2024, 09:00 - A: x\n05/01/2024, 09:00 - B: y\n", date_order="MDY")
    assert list(r.df["sender"]) == ["B"]
    assert r.report.skipped_records == 1


def test_no_timezone_is_invented():
    r = parse_chat("05/01/2024, 09:00 - A: x\n")
    assert r.df["timestamp"].dt.tz is None


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------
def test_stable_chronological_sort_keeps_ties_in_file_order():
    chat = "05/01/2024, 09:05 - A: third\n05/01/2024, 09:00 - B: first\n05/01/2024, 09:00 - C: second (tie)\n"
    r = parse_chat(chat)
    assert list(r.df["text"]) == ["first", "second (tie)", "third"]
    assert r.report.out_of_order == 1


# ---------------------------------------------------------------------------
# Edge cases and limits
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["", "   \n\n", "hello world\nno timestamps here\n"])
def test_empty_or_unsupported_input_raises_safe_error(text):
    with pytest.raises(ParseError) as exc:
        parse_chat(text)
    assert "hello" not in str(exc.value)  # never echoes content


def test_preamble_lines_reported():
    r = parse_chat("garbage line\n05/01/2024, 09:00 - A: x\n")
    assert r.report.preamble_lines == 1
    assert len(r.df) == 1


def test_system_only_and_media_only_exports():
    sys_only = parse_chat("05/01/2024, 09:00 - Messages and calls are end-to-end encrypted.\n")
    assert sys_only.report.type_counts["system"] == 1
    assert any("only system" in w for w in sys_only.report.warnings)
    media_only = parse_chat("05/01/2024, 09:00 - A: <Media omitted>\n")
    assert any("no text messages" in w for w in media_only.report.warnings)


def test_message_limit_keeps_most_recent(monkeypatch):
    monkeypatch.setattr(config, "MAX_MESSAGES", 3)
    chat = "".join(f"05/01/2024, 09:0{i} - A: m{i}\n" for i in range(5))
    r = parse_chat(chat)
    assert list(r.df["text"]) == ["m2", "m3", "m4"]
    assert r.report.truncated == 2


def test_read_upload_rejects_oversize_and_binary(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(ParseError, match="limit"):
        read_upload(b"x" * 11, "a.txt")
    monkeypatch.undo()
    with pytest.raises(ParseError, match="binary"):
        read_upload(b"\x00\x01\x02abc", "a.txt")
    with pytest.raises(ParseError, match="empty"):
        read_upload(b"", "a.txt")


def test_read_upload_invalid_utf8_is_replaced_with_warning():
    text, warnings = read_upload(b"05/01/2024, 09:00 - A: caf\xe9\n", "a.txt")
    assert "�" in text
    assert warnings and "UTF-8" in warnings[0]


def test_read_upload_utf16():
    raw = "05/01/2024, 09:00 - A: hi\n".encode("utf-16")
    text, _ = read_upload(raw, "a.txt")
    assert parse_chat(text).df.loc[0, "text"] == "hi"


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_read_upload_zip_prefers_chat_txt():
    data = _zip({"IMG-1.jpg": b"\xff\xd8", "_chat.txt": b"[05/01/2024, 09:00:00] A: hi\n"})
    text, _ = read_upload(data, "WhatsApp Chat.zip")
    assert parse_chat(text).df.loc[0, "sender"] == "A"


def test_read_upload_zip_bomb_guard(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    data = _zip({"_chat.txt": b"a" * 50_000})  # compresses well below 1000 bytes
    assert len(data) < 1000
    with pytest.raises(ParseError, match="limit"):
        read_upload(data, "c.zip")


def test_read_upload_zip_without_text():
    with pytest.raises(ParseError, match="does not contain"):
        read_upload(_zip({"a.jpg": b"\xff"}), "c.zip")


def test_timestamp_like_text_inside_message_keeps_alignment():
    # The original parser crashed here ("All arrays must be of the same length").
    chat = (
        "1/5/24, 9:01 AM - Alice: meet at 1/6/24, 10:00 AM ok?\n"
        "1/5/24, 9:02 AM - Bob: sure\n"
        "see 1/7/24, 11:00 AM - also fine\n"
        "1/13/24, 9:03 AM - Alice: done\n"
    )
    r = parse_chat(chat)
    assert list(r.df["sender"]) == ["Alice", "Bob", "Alice"]
    assert r.df.loc[0, "text"] == "meet at 1/6/24, 10:00 AM ok?"
    assert r.df.loc[1, "text"] == "sure\nsee 1/7/24, 11:00 AM - also fine"
    assert r.df.loc[1, "timestamp"] == datetime(2024, 1, 5, 9, 2)
