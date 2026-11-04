"""Pure statistics on a parsed chat DataFrame.

Every function takes a DataFrame produced by :func:`wa_analyzer.parsing.parse_chat`
(or a filtered subset), never mutates it, and returns the same type for empty
input as for non-empty input.

Definitions
-----------
- **Message**: a participant entry (text, media placeholder or deleted marker).
  System notices (joins, encryption notice, subject changes …) are counted
  separately as *system events* and never as messages.
- **Active day**: a calendar date with at least one message.
- **Calendar span**: days from the first to the last message, inclusive.
- **Session**: consecutive messages in which no gap exceeds the idle threshold
  (default 60 minutes). The *session starter* is the sender of the first
  message after an idle gap. This describes timing, not intent.
- **Response gap (estimate)**: time between a message and the next message by a
  *different* sender within the same session, computed on the complete
  chronological conversation before any participant filter. In group chats the
  next message is not necessarily a reply, so these are estimates.
- **Silence gap**: a gap between consecutive messages (all participants) that
  exceeds a threshold. It describes when the chat was quiet, not who was
  responsible.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from . import text as textlib

USER_TYPES = ("text", "media", "deleted")
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
DEFAULT_IDLE_MINUTES = 60


@dataclass(frozen=True)
class Filters:
    participants: tuple[str, ...] | None = None  # None = everyone
    start: date | None = None
    end: date | None = None

    def key(self) -> tuple:
        return (self.participants, self.start, self.end)


@dataclass(frozen=True)
class Overview:
    messages: int
    text_messages: int
    media_messages: int
    deleted_messages: int
    system_events: int
    words: int
    links: int
    emoji: int
    participants: int
    first_message: datetime | None
    last_message: datetime | None
    calendar_days: int
    active_days: int
    avg_per_active_day: float | None
    avg_per_calendar_day: float | None


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------
def user_messages(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["message_type"].isin(USER_TYPES)]


def participants(df: pd.DataFrame) -> list[str]:
    """Participants ordered by message count (desc), then name."""
    msgs = user_messages(df)
    if msgs.empty:
        return []
    counts = msgs.groupby("sender").size().reset_index(name="n")
    counts = counts.sort_values(["n", "sender"], ascending=[False, True], kind="mergesort")
    return counts["sender"].astype(str).tolist()


def _date_mask(df: pd.DataFrame, start: date | None, end: date | None) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df["timestamp"] >= pd.Timestamp(start)
    if end is not None:
        mask &= df["timestamp"] < pd.Timestamp(end) + pd.Timedelta(days=1)
    return mask


def apply_filters(df: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    """Rows inside the date range (inclusive) from the selected participants.

    System events have no sender: they are kept only when no participant
    filter is active.
    """
    mask = _date_mask(df, filters.start, filters.end)
    if filters.participants is not None:
        mask &= df["sender"].isin(filters.participants) & df["message_type"].isin(USER_TYPES)
    return df[mask]


# ---------------------------------------------------------------------------
# Overview and per-participant counts
# ---------------------------------------------------------------------------
def overview(df: pd.DataFrame) -> Overview:
    msgs = user_messages(df)
    texts = msgs.loc[msgs["message_type"] == "text", "text"]
    n = len(msgs)
    if n:
        first = msgs["timestamp"].min().to_pydatetime()
        last = msgs["timestamp"].max().to_pydatetime()
        calendar_days = (last.date() - first.date()).days + 1
        active_days = int(msgs["timestamp"].dt.normalize().nunique())
    else:
        first = last = None
        calendar_days = active_days = 0
    return Overview(
        messages=n,
        text_messages=int((msgs["message_type"] == "text").sum()),
        media_messages=int((msgs["message_type"] == "media").sum()),
        deleted_messages=int((msgs["message_type"] == "deleted").sum()),
        system_events=int((df["message_type"] == "system").sum()),
        words=int(sum(textlib.count_words(t) for t in texts)),
        links=int(sum(textlib.count_links(t) for t in texts)),
        emoji=int(sum(len(textlib.extract_emoji(t)) for t in texts)),
        participants=int(msgs["sender"].nunique()),
        first_message=first,
        last_message=last,
        calendar_days=calendar_days,
        active_days=active_days,
        avg_per_active_day=(n / active_days) if active_days else None,
        avg_per_calendar_day=(n / calendar_days) if calendar_days else None,
    )


PARTICIPANT_COLUMNS = ["participant", "messages", "text", "media", "deleted", "words", "links", "emoji", "share_pct"]


def participant_summary(df: pd.DataFrame) -> pd.DataFrame:
    msgs = user_messages(df)
    if msgs.empty:
        return pd.DataFrame(columns=PARTICIPANT_COLUMNS)
    is_text = msgs["message_type"] == "text"
    text_series = msgs["text"].where(is_text, "")
    frame = pd.DataFrame(
        {
            "participant": msgs["sender"].astype(str),
            "text": is_text.astype(int),
            "media": (msgs["message_type"] == "media").astype(int),
            "deleted": (msgs["message_type"] == "deleted").astype(int),
            "words": text_series.map(textlib.count_words),
            "links": text_series.map(textlib.count_links),
            "emoji": text_series.map(lambda t: len(textlib.extract_emoji(t))),
        }
    )
    out = frame.groupby("participant", sort=False).sum().reset_index()
    out.insert(1, "messages", out["text"] + out["media"] + out["deleted"])
    out["share_pct"] = (100 * out["messages"] / out["messages"].sum()).round(1)
    out = out.sort_values(["messages", "participant"], ascending=[False, True], kind="mergesort")
    return out[PARTICIPANT_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------
def daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Messages per calendar day, including days with zero messages."""
    msgs = user_messages(df)
    if msgs.empty:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "messages": pd.Series(dtype="int64")})
    days = msgs["timestamp"].dt.normalize()
    counts = days.value_counts().sort_index()
    full = pd.date_range(days.min(), days.max(), freq="D")
    counts = counts.reindex(full, fill_value=0)
    return pd.DataFrame({"date": full, "messages": counts.to_numpy(dtype="int64")})


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Messages per calendar month (year-aware, chronological, zero-filled)."""
    msgs = user_messages(df)
    if msgs.empty:
        return pd.DataFrame(
            {
                "month": pd.Series(dtype="datetime64[ns]"),
                "label": pd.Series(dtype=object),
                "messages": pd.Series(dtype="int64"),
            }
        )
    periods = msgs["timestamp"].dt.to_period("M")
    counts = periods.value_counts()
    full = pd.period_range(periods.min(), periods.max(), freq="M")
    counts = counts.reindex(full, fill_value=0)
    return pd.DataFrame(
        {
            "month": full.to_timestamp(),
            "label": [p.strftime("%Y-%m") for p in full],
            "messages": counts.to_numpy(dtype="int64"),
        }
    )


def weekly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Messages per week (weeks start on Monday; year-aware)."""
    msgs = user_messages(df)
    if msgs.empty:
        return pd.DataFrame({"week_start": pd.Series(dtype="datetime64[ns]"), "messages": pd.Series(dtype="int64")})
    periods = msgs["timestamp"].dt.to_period("W-SUN")
    counts = periods.value_counts()
    full = pd.period_range(periods.min(), periods.max(), freq="W-SUN")
    counts = counts.reindex(full, fill_value=0)
    return pd.DataFrame({"week_start": full.start_time, "messages": counts.to_numpy(dtype="int64")})


def hourly_counts(df: pd.DataFrame) -> pd.DataFrame:
    msgs = user_messages(df)
    counts = msgs["timestamp"].dt.hour.value_counts().reindex(range(24), fill_value=0)
    return pd.DataFrame({"hour": list(range(24)), "messages": counts.to_numpy(dtype="int64")})


def weekday_counts(df: pd.DataFrame) -> pd.DataFrame:
    msgs = user_messages(df)
    counts = msgs["timestamp"].dt.day_of_week.value_counts().reindex(range(7), fill_value=0)
    return pd.DataFrame({"weekday": list(WEEKDAYS), "messages": counts.to_numpy(dtype="int64")})


def weekday_hour_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """7 × 24 matrix (Monday…Sunday × hour 0…23) of message counts."""
    msgs = user_messages(df)
    matrix = pd.crosstab(msgs["timestamp"].dt.day_of_week, msgs["timestamp"].dt.hour) if len(msgs) else pd.DataFrame()
    matrix = matrix.reindex(index=range(7), columns=range(24), fill_value=0).fillna(0).astype("int64")
    matrix.index = list(WEEKDAYS)
    matrix.columns = list(range(24))
    return matrix


# ---------------------------------------------------------------------------
# Conversation timing (computed on the full chronological conversation)
# ---------------------------------------------------------------------------
def timeline(full_df: pd.DataFrame, idle_minutes: int = DEFAULT_IDLE_MINUTES) -> pd.DataFrame:
    """Participant messages in order with gap/session annotations.

    Uses the *complete* chat (all participants, all dates) so that gaps and
    sessions are never distorted by filters.
    """
    msgs = user_messages(full_df)[["timestamp", "sender"]].reset_index(drop=True)
    prev_ts = msgs["timestamp"].shift(1)
    prev_sender = msgs["sender"].shift(1)
    gap_min = (msgs["timestamp"] - prev_ts).dt.total_seconds() / 60.0
    new_session = gap_min.isna() | (gap_min > idle_minutes)
    return msgs.assign(
        prev_sender=prev_sender,
        gap_minutes=gap_min,
        sender_changed=prev_sender.notna() & (msgs["sender"] != prev_sender),
        session_start=new_session,
        session_id=new_session.cumsum().astype("int64"),
    )


def _in_range(ts: pd.Series, filters: Filters) -> pd.Series:
    mask = pd.Series(True, index=ts.index)
    if filters.start is not None:
        mask &= ts >= pd.Timestamp(filters.start)
    if filters.end is not None:
        mask &= ts < pd.Timestamp(filters.end) + pd.Timedelta(days=1)
    return mask


RESPONSE_COLUMNS = [
    "participant",
    "reply_gaps_counted",
    "median_reply_gap_min",
    "mean_reply_gap_min",
    "followups_counted",
    "median_followup_gap_min",
]


def response_gaps(full_df: pd.DataFrame, filters: Filters, idle_minutes: int = DEFAULT_IDLE_MINUTES) -> pd.DataFrame:
    """Per-participant reply-gap estimates and same-sender follow-up gaps.

    A *reply gap* is counted for a message whose sender differs from the
    previous message's sender within the same session. A *follow-up gap* is
    the gap before a consecutive message by the same sender within a session.
    Rows are filtered (participant, date of the later message) only after the
    gaps are computed on the full conversation.
    """
    tl = timeline(full_df, idle_minutes)
    if tl.empty:
        return pd.DataFrame(columns=RESPONSE_COLUMNS)
    within = ~tl["session_start"]
    keep = within & _in_range(tl["timestamp"], filters)
    if filters.participants is not None:
        keep &= tl["sender"].isin(filters.participants)
    sel = tl[keep]
    replies = sel[sel["sender_changed"]]
    followups = sel[~sel["sender_changed"]]
    people = filters.participants if filters.participants is not None else participants(full_df)
    rows = []
    for person in people:
        r = replies.loc[replies["sender"] == person, "gap_minutes"]
        f = followups.loc[followups["sender"] == person, "gap_minutes"]
        rows.append(
            {
                "participant": person,
                "reply_gaps_counted": int(len(r)),
                "median_reply_gap_min": round(float(r.median()), 1) if len(r) else None,
                "mean_reply_gap_min": round(float(r.mean()), 1) if len(r) else None,
                "followups_counted": int(len(f)),
                "median_followup_gap_min": round(float(f.median()), 1) if len(f) else None,
            }
        )
    return pd.DataFrame(rows, columns=RESPONSE_COLUMNS)


SESSION_COLUMNS = ["session_id", "start", "end", "starter", "messages", "participants", "duration_min"]
STARTER_COLUMNS = ["participant", "sessions_started"]


def sessions(
    full_df: pd.DataFrame, filters: Filters, idle_minutes: int = DEFAULT_IDLE_MINUTES
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sessions whose first message falls in the date range and that include a
    selected participant, plus counts of sessions started per participant."""
    tl = timeline(full_df, idle_minutes)
    if tl.empty:
        return pd.DataFrame(columns=SESSION_COLUMNS), pd.DataFrame(columns=STARTER_COLUMNS)
    grouped = tl.groupby("session_id", sort=True)
    sess = pd.DataFrame(
        {
            "start": grouped["timestamp"].min(),
            "end": grouped["timestamp"].max(),
            "starter": grouped["sender"].first(),
            "messages": grouped.size(),
            "participants": grouped["sender"].nunique(),
        }
    ).reset_index()
    sess["duration_min"] = ((sess["end"] - sess["start"]).dt.total_seconds() / 60).round(1)
    keep = _in_range(sess["start"], filters)
    if filters.participants is not None:
        involved = tl[tl["sender"].isin(filters.participants)]["session_id"].unique()
        keep &= sess["session_id"].isin(involved)
    sess = sess[keep][SESSION_COLUMNS].reset_index(drop=True)
    people = filters.participants if filters.participants is not None else participants(full_df)
    started = sess["starter"].value_counts()
    starters = pd.DataFrame(
        {"participant": list(people), "sessions_started": [int(started.get(p, 0)) for p in people]},
        columns=STARTER_COLUMNS,
    )
    starters = starters.sort_values("sessions_started", ascending=False, kind="mergesort").reset_index(drop=True)
    return sess, starters


SILENCE_COLUMNS = ["gap_start", "gap_end", "gap_hours", "last_sender_before", "first_sender_after"]


def silence_gaps(full_df: pd.DataFrame, filters: Filters, min_hours: float = 24.0, top_n: int = 10) -> pd.DataFrame:
    """Longest quiet periods in the complete conversation (all participants).

    Gaps whose end falls inside the date range are returned. Participant
    filters do not apply: silence is a property of the conversation.
    """
    tl = timeline(full_df)
    if len(tl) < 2:
        return pd.DataFrame(columns=SILENCE_COLUMNS)
    hours = tl["gap_minutes"] / 60.0
    mask = (hours >= min_hours) & _in_range(tl["timestamp"], filters)
    sel = tl[mask]
    out = pd.DataFrame(
        {
            "gap_start": sel["timestamp"] - pd.to_timedelta(sel["gap_minutes"], unit="min"),
            "gap_end": sel["timestamp"],
            "gap_hours": (sel["gap_minutes"] / 60.0).round(1),
            "last_sender_before": sel["prev_sender"],
            "first_sender_after": sel["sender"],
        },
        columns=SILENCE_COLUMNS,
    )
    return out.sort_values("gap_hours", ascending=False, kind="mergesort").head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Words and emoji
# ---------------------------------------------------------------------------
def _text_series(df: pd.DataFrame) -> Iterable[str]:
    return df.loc[df["message_type"] == "text", "text"].astype(str)


def top_words(df: pd.DataFrame, n: int = 20, stopwords: frozenset[str] = textlib.STOPWORDS) -> pd.DataFrame:
    counter = textlib.word_counter(_text_series(df), stopwords=stopwords)
    return pd.DataFrame(counter.most_common(n), columns=["word", "count"])


def word_frequencies(df: pd.DataFrame, max_words: int = 150) -> dict[str, int]:
    return dict(textlib.word_counter(_text_series(df)).most_common(max_words))


def top_emoji(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    counter = textlib.emoji_counter(_text_series(df))
    return pd.DataFrame(counter.most_common(n), columns=["emoji", "count"])


def selected_participants(filters: Filters, everyone: Sequence[str]) -> list[str]:
    return list(filters.participants) if filters.participants is not None else list(everyone)
