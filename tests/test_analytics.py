"""Analytics tests against hand-calculated values on a synthetic chat.

Timeline of participant messages in FIXTURE (system notice excluded):

    #  sender  time              type     gap(min)  session
    0  A       01 Mar 09:00      text     -         1 (start)
    1  B       01 Mar 09:10      text     10        1   reply gap B=10
    2  B       01 Mar 09:15      media    5         1   follow-up B=5
    3  A       01 Mar 09:45      text     30        1   reply gap A=30
    4  C       03 Mar 20:00      text     3495      2 (start; 58.25 h silence)
    5  A       03 Mar 20:30      deleted  30        2   reply gap A=30
    6  B       03 Mar 22:00      text     90        3 (start, >60 min idle)
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from wa_analyzer import analytics as an
from wa_analyzer.parsing import parse_chat

FIXTURE = (
    "01/03/2024, 09:00 - Messages and calls are end-to-end encrypted.\n"
    "01/03/2024, 09:00 - A: hello there 😀\n"
    "01/03/2024, 09:10 - B: hi A https://x.org\n"
    "01/03/2024, 09:15 - B: <Media omitted>\n"
    "01/03/2024, 09:45 - A: ok 👍🏽👍🏽\n"
    "03/03/2024, 20:00 - C: back again\n"
    "03/03/2024, 20:30 - A: This message was deleted\n"
    "03/03/2024, 22:00 - B: 🇮🇳 late\n"
)
ALL = an.Filters()


@pytest.fixture()
def df():
    return parse_chat(FIXTURE, date_order="DMY").df


def test_overview_hand_calculated(df):
    ov = an.overview(df)
    assert (ov.messages, ov.text_messages, ov.media_messages, ov.deleted_messages, ov.system_events) == (7, 5, 1, 1, 1)
    assert ov.words == 8  # hello there(2) hi A(2) ok(1) back again(2) late(1); link not a word
    assert ov.links == 1
    assert ov.emoji == 4  # 😀, 👍🏽 ×2, 🇮🇳 — skin tone and flag count once each
    assert ov.participants == 3
    assert ov.first_message == datetime(2024, 3, 1, 9, 0)
    assert ov.last_message == datetime(2024, 3, 3, 22, 0)
    assert ov.calendar_days == 3
    assert ov.active_days == 2
    assert ov.avg_per_active_day == pytest.approx(3.5)
    assert ov.avg_per_calendar_day == pytest.approx(7 / 3)


def test_participant_summary(df):
    ps = an.participant_summary(df)
    assert ps["participant"].tolist() == ["A", "B", "C"]
    assert ps["messages"].tolist() == [3, 3, 1]
    assert ps.set_index("participant").loc["B", ["text", "media", "deleted", "links"]].tolist() == [2, 1, 0, 1]
    assert ps["share_pct"].tolist() == [42.9, 42.9, 14.3]


def test_daily_hourly_weekday(df):
    daily = an.daily_counts(df)
    assert daily["date"].dt.date.tolist() == [date(2024, 3, 1), date(2024, 3, 2), date(2024, 3, 3)]
    assert daily["messages"].tolist() == [4, 0, 3]
    hourly = an.hourly_counts(df).set_index("hour")["messages"]
    assert (hourly[9], hourly[20], hourly[22], hourly.sum()) == (4, 2, 1, 7)
    weekday = an.weekday_counts(df)
    assert weekday["weekday"].tolist() == list(an.WEEKDAYS)  # Monday first, real names
    assert weekday.set_index("weekday").loc[["Friday", "Sunday"], "messages"].tolist() == [4, 3]
    matrix = an.weekday_hour_matrix(df)
    assert matrix.shape == (7, 24)
    assert matrix.loc["Friday", 9] == 4 and matrix.to_numpy().sum() == 7


def test_monthly_and_weekly_are_year_aware_and_chronological():
    chat = (
        "15/12/2023, 10:00 - A: dec\n"
        "20/02/2024, 10:00 - A: feb\n"
        "21/02/2024, 10:00 - B: feb2\n"
        "16/12/2024, 10:00 - B: dec next year\n"
    )
    df = parse_chat(chat).df
    monthly = an.monthly_counts(df)
    assert monthly["label"].tolist()[:3] == ["2023-12", "2024-01", "2024-02"]
    assert monthly["label"].tolist()[-1] == "2024-12"
    assert len(monthly) == 13
    assert monthly.set_index("label").loc[["2023-12", "2024-01", "2024-02", "2024-12"], "messages"].tolist() == [1, 0, 2, 1]
    weekly = an.weekly_counts(df)
    assert weekly["week_start"].is_monotonic_increasing
    assert weekly["messages"].sum() == 4
    assert (weekly["week_start"].dt.day_of_week == 0).all()  # Mondays


def test_response_gaps_full_conversation_then_filter(df):
    rg = an.response_gaps(df, ALL).set_index("participant")
    assert rg.loc["A", "reply_gaps_counted"] == 2
    assert rg.loc["A", "median_reply_gap_min"] == 30.0
    assert rg.loc["B", "reply_gaps_counted"] == 1
    assert rg.loc["B", "median_reply_gap_min"] == 10.0
    assert rg.loc["B", "followups_counted"] == 1
    assert rg.loc["B", "median_followup_gap_min"] == 5.0
    assert rg.loc["C", "reply_gaps_counted"] == 0  # C's message started a session
    assert pd.isna(rg.loc["C", "median_reply_gap_min"])

    # Filtering to A must not turn A's replies into A-to-A gaps.
    only_a = an.response_gaps(df, an.Filters(participants=("A",)))
    assert only_a["participant"].tolist() == ["A"]
    assert only_a.loc[0, "reply_gaps_counted"] == 2
    assert only_a.loc[0, "median_reply_gap_min"] == 30.0

    # Date filter applies to the later message; the earlier one may be outside the range.
    later = an.response_gaps(df, an.Filters(start=date(2024, 3, 3))).set_index("participant")
    assert later.loc["A", "reply_gaps_counted"] == 1
    assert later.loc["B", "reply_gaps_counted"] == 0


def test_sessions_idle_rule(df):
    sess, starters = an.sessions(df, ALL, idle_minutes=60)
    assert sess["starter"].tolist() == ["A", "C", "B"]
    assert sess["messages"].tolist() == [4, 2, 1]
    assert sess["duration_min"].tolist() == [45.0, 30.0, 0.0]
    assert dict(zip(starters["participant"], starters["sessions_started"])) == {"A": 1, "B": 1, "C": 1}
    # A larger idle threshold merges the 90-minute gap.
    sess3h, _ = an.sessions(df, ALL, idle_minutes=180)
    assert sess3h["messages"].tolist() == [4, 3]
    # Date filter keeps sessions that start in range.
    later, _ = an.sessions(df, an.Filters(start=date(2024, 3, 3)))
    assert later["starter"].tolist() == ["C", "B"]


def test_silence_gaps_from_complete_timeline(df):
    gaps = an.silence_gaps(df, ALL, min_hours=24)
    assert len(gaps) == 1
    row = gaps.iloc[0]
    assert row["gap_hours"] == pytest.approx(58.25, abs=0.05)  # 3495 min, rounded to 0.1 h
    assert row["gap_start"] == pd.Timestamp("2024-03-01 09:45")
    assert row["gap_end"] == pd.Timestamp("2024-03-03 20:00")
    assert (row["last_sender_before"], row["first_sender_after"]) == ("A", "C")
    # A participant filter does not change conversation-level silences.
    assert len(an.silence_gaps(df, an.Filters(participants=("B",)), min_hours=24)) == 1


def test_filters_apply_consistently(df):
    only_b = an.apply_filters(df, an.Filters(participants=("B",)))
    assert set(only_b["sender"]) == {"B"}
    assert (only_b["message_type"] != "system").all()
    ov = an.overview(only_b)
    assert (ov.messages, ov.media_messages, ov.links, ov.system_events) == (3, 1, 1, 0)
    first_day = an.apply_filters(df, an.Filters(start=date(2024, 3, 1), end=date(2024, 3, 1)))
    assert an.overview(first_day).messages == 4
    assert an.daily_counts(first_day)["messages"].tolist() == [4]
    assert an.top_words(first_day)["word"].tolist().count("late") == 0


def test_words_and_emoji(df):
    words = an.top_words(df)
    assert set(words["word"]) == {"hello", "hi", "back", "late"}  # 'there', 'ok', 'again' are stop words
    emo = an.top_emoji(df)
    assert dict(zip(emo["emoji"], emo["count"])) == {"👍🏽": 2, "😀": 1, "🇮🇳": 1}


def test_functions_do_not_mutate_input(df):
    before = df.copy(deep=True)
    an.overview(df)
    an.participant_summary(df)
    an.daily_counts(df)
    an.monthly_counts(df)
    an.weekly_counts(df)
    an.weekday_hour_matrix(df)
    an.response_gaps(df, ALL)
    an.sessions(df, ALL)
    an.silence_gaps(df, ALL)
    an.top_words(df)
    an.top_emoji(df)
    an.word_frequencies(df)
    pd.testing.assert_frame_equal(df, before)


def test_empty_inputs_return_consistent_types(df):
    empty = an.apply_filters(df, an.Filters(participants=("nobody",)))
    assert empty.empty
    ov = an.overview(empty)
    assert ov.messages == 0 and ov.avg_per_active_day is None and ov.first_message is None
    assert list(an.participant_summary(empty).columns) == an.PARTICIPANT_COLUMNS
    assert an.daily_counts(empty).empty and list(an.daily_counts(empty).columns) == ["date", "messages"]
    assert an.monthly_counts(empty).empty
    assert an.weekly_counts(empty).empty
    assert an.hourly_counts(empty)["messages"].sum() == 0
    assert an.weekday_hour_matrix(empty).shape == (7, 24)
    assert an.top_words(empty).empty and list(an.top_words(empty).columns) == ["word", "count"]
    assert an.top_emoji(empty).empty
    assert an.word_frequencies(empty) == {}
    assert an.response_gaps(empty, ALL).empty
    sess, starters = an.sessions(empty, ALL)
    assert sess.empty and starters.empty
    assert an.silence_gaps(empty, ALL).empty


def test_system_only_chat():
    df = parse_chat("05/01/2024, 09:00 - Messages and calls are end-to-end encrypted.\n").df
    assert an.participants(df) == []
    ov = an.overview(df)
    assert ov.messages == 0 and ov.system_events == 1
