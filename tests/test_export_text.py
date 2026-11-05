"""Export (CSV formula protection, report) and text-helper tests."""

from __future__ import annotations

import csv
import io
import zipfile

import pandas as pd
import pytest

from wa_analyzer import analytics as an
from wa_analyzer import export as ex
from wa_analyzer import text as tx
from wa_analyzer.parsing import parse_chat
from wa_analyzer.sample import SAMPLE_CHAT


@pytest.mark.parametrize(
    "value,expected",
    [
        ('=HYPERLINK("http://x")', '\'=HYPERLINK("http://x")'),
        ("+1 555 0100", "'+1 555 0100"),
        ("-2+3", "'-2+3"),
        ("@SUM(A1)", "'@SUM(A1)"),
        ("\t=1", "'\t=1"),
        ("\r=1", "'\r=1"),
        ("  =1", "'  =1"),
        ("Alice", "Alice"),
        ("", ""),
        (42, 42),
    ],
)
def test_csv_safe(value, expected):
    assert ex.csv_safe(value) == expected


def test_to_safe_csv_round_trip_neutralises_names():
    table = pd.DataFrame({"participant": ["=cmd|' /C calc'!A0", "Bob"], "messages": [3, 1]})
    raw = ex.to_safe_csv(table).decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(raw)))
    assert rows[0] == ["participant", "messages"]
    assert rows[1][0].startswith("'=")
    assert rows[2] == ["Bob", "1"]


def test_export_zip_and_report_exclude_message_text():
    result = parse_chat(SAMPLE_CHAT)
    df = result.df
    ov = an.overview(df)
    parts = an.participant_summary(df)
    report = ex.build_report(result.report, ov, parts, "All participants, all dates")
    data = ex.build_zip(
        {"overview": ex.overview_table(ov), "participants": parts, "daily": an.daily_counts(df)}, report
    )
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        blob = b"".join(zf.read(n) for n in names).decode("utf-8-sig", errors="replace")
    assert names == {"overview.csv", "participants.csv", "daily.csv", "report.md"}
    for snippet in ("north gate", "Bring water", "check the bus times", "permits"):
        assert snippet not in blob
    assert "Messages (text + media + deleted)" in blob
    assert f"{ov.messages:,}" in report


def test_overview_table_values_match_overview():
    df = parse_chat(SAMPLE_CHAT).df
    ov = an.overview(df)
    table = ex.overview_table(ov).set_index("metric")["value"]
    assert table["Messages (text + media + deleted)"] == str(ov.messages)
    assert table["Active days (≥1 message)"] == str(ov.active_days)


def test_pseudonym_map():
    m = ex.pseudonym_map([f"n{i}" for i in range(28)])
    assert m["n0"] == "Participant A"
    assert m["n25"] == "Participant Z"
    assert m["n26"] == "Participant AA"
    assert len(set(m.values())) == 28


def test_text_helpers():
    assert tx.count_words("hello, world! 👍 https://x.org 10:30") == 3
    assert tx.count_links("see https://a.b/c and www.x.org") == 2
    assert tx.tokenize("Hello, WORLD! नमस्ते। કેમ છો?") == ["hello", "world", "नमस्ते", "કેમ", "છો"]
    assert tx.extract_emoji("👍🏽🇮🇳👨‍👩‍👧 © ❤️") == ["👍🏽", "🇮🇳", "👨‍👩‍👧", "❤️"]
    assert tx.extract_emoji("") == []
    assert tx.word_counter(["the cat the dog cat"]).most_common(1) == [("cat", 2)]
