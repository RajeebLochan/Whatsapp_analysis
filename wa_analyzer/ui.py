"""Streamlit user interface.

Privacy model
-------------
- Uploaded bytes, parsed DataFrames, analytics, sentiment results and AI
  summaries live only in ``st.session_state`` for the current browser session.
  Nothing chat-specific is placed in ``st.cache_data``/``st.cache_resource``
  (those are shared by every user of the server). The only shared resource is
  the immutable sentiment model.
- Nothing is written to disk by the app. "Clear chat and results" removes the
  current session's data (other sessions are unaffected).
- External requests happen only when the user clicks "Generate AI summary"
  after ticking the consent box, or when the optional sentiment model is
  downloaded from Hugging Face on first use.
"""

from __future__ import annotations

import hashlib
import html
import logging
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import ai_summary, analytics, config, export, parsing, sentiment
from .sample import SAMPLE_CHAT

logger = logging.getLogger(__name__)

# Colours: restrained green accent; sentiment uses a blue/grey/red diverging
# scheme (not green/red) so it remains readable with red-green colour blindness.
ACCENT = "#128C7E"
ACCENT_LIGHT = "#5ED3A2"
SENTIMENT_COLORS = {"negative": "#E34948", "neutral": "#8F8D87", "positive": "#2A78D6"}
HEATMAP_SCALE = [[0.0, "#EDF8F1"], [0.5, "#5ED3A2"], [1.0, "#0B5D47"]]

STATE_PREFIX = "wa_"
MEMO_SIZE = 12
IDLE_OPTIONS = {30: "30 minutes", 60: "1 hour", 180: "3 hours", 360: "6 hours", 720: "12 hours", 1440: "24 hours"}

CSS = """
<style>
  .block-container {max-width: 1200px; padding-top: 2rem;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
  .wa-muted {opacity: 0.8; font-size: 0.9rem;}
</style>
"""


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def _ss() -> Any:
    return st.session_state


def init_state() -> None:
    ss = _ss()
    ss.setdefault("wa_uploader_gen", 0)
    ss.setdefault("wa_sample_loaded", False)
    ss.setdefault("wa_memo", OrderedDict())
    ss.setdefault("wa_sentiment_runs", OrderedDict())
    ss.setdefault("wa_ai_results", OrderedDict())
    ss.setdefault("wa_ai_requests", 0)
    ss.setdefault("wa_current_sha", None)


def clear_session() -> None:
    """Remove this session's chat, derived data and AI results."""
    ss = _ss()
    gen = ss.get("wa_uploader_gen", 0)
    for key in [k for k in list(ss.keys()) if str(k).startswith(STATE_PREFIX)]:
        del ss[key]
    ss["wa_uploader_gen"] = gen + 1  # a fresh uploader widget drops the old file
    ss["wa_cleared"] = True


def load_sample() -> None:
    clear_session()
    _ss()["wa_cleared"] = False
    _ss()["wa_sample_loaded"] = True


def _drop_stale(sha: str) -> None:
    """Invalidate everything derived from a previous upload."""
    ss = _ss()
    if ss.get("wa_current_sha") not in (None, sha):
        ss["wa_memo"] = OrderedDict()
        ss["wa_sentiment_runs"] = OrderedDict()
        ss["wa_ai_results"] = OrderedDict()
        for key in [k for k in list(ss.keys()) if str(k).startswith(("wa_parsed", "wa_f_", "wa_ai_consent"))]:
            del ss[key]
    ss["wa_current_sha"] = sha


def memo(key: tuple, fn: Callable[[], Any]) -> Any:
    """Session-scoped memoisation (bounded LRU). Never shared across users."""
    store: OrderedDict = _ss()["wa_memo"]
    if key in store:
        store.move_to_end(key)
        return store[key]
    value = fn()
    store[key] = value
    while len(store) > MEMO_SIZE:
        store.popitem(last=False)
    return value


def _lru_put(store: OrderedDict, key: Any, value: Any, size: int = 4) -> None:
    store[key] = value
    store.move_to_end(key)
    while len(store) > size:
        store.popitem(last=False)


@contextmanager
def safe_section(name: str):
    """Show a generic error instead of a raw traceback; log only the type."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - keep other sections working
        # Streamlit's rerun/stop signals derive from BaseException and pass through.
        logger.error("Section %s failed: %s", name, type(exc).__name__)
        st.error(f"The {name} section could not be displayed for this selection.")


def safe_label(value: object) -> str:
    """Escape user-controlled text for Plotly (which renders a subset of HTML)."""
    return html.escape(str(value), quote=True)


def md_escape(value: object) -> str:
    return ai_summary.escape_markdown(str(value))


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def show_chart(fig: go.Figure, height: int = 360) -> None:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        hoverlabel=dict(namelength=-1),
        bargap=0.25,
    )
    st.plotly_chart(fig, width="stretch", theme="streamlit", config={"displaylogo": False, "displayModeBar": False})


def hbar(labels: list[str], values: list[float], title: str, value_name: str, color: str = ACCENT) -> go.Figure:
    labels = [safe_label(lbl) for lbl in labels]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=color, cornerradius=4),
            hovertemplate=f"%{{y}}<br>{value_name}: %{{x:,}}<extra></extra>",
        )
    )
    fig.update_layout(title=title, yaxis=dict(autorange="reversed", automargin=True), xaxis_title=value_name)
    return fig


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def header() -> None:
    st.title("💬 WhatsApp Chat Analyzer")
    st.caption(
        "Statistics for an exported WhatsApp chat: activity, participants, words and emoji, "
        "with optional sentiment and an optional, opt-in AI summary."
    )


PRIVACY_NOTE = (
    "**Privacy:** this app runs on a server. When it is hosted (for example on Streamlit Community Cloud "
    "or Azure), your file is uploaded to and processed on that server — it is **not** processed only in "
    "your browser. The app keeps the chat in memory for your session only, does not write it to disk "
    "or a database, and never sends it anywhere unless you explicitly request an AI summary. "
    "Use **Clear chat and results** when you are done; data also disappears when the session ends. "
    "Hosting providers may keep their own access logs."
)


def upload_step() -> tuple[str, bytes, str] | None:
    """Return ``(sha256, raw_bytes, source_label)`` or None."""
    st.subheader("1 · Upload a chat export")
    with st.container(border=True):
        st.markdown(PRIVACY_NOTE)
        col_up, col_side = st.columns([3, 2], gap="medium")
        with col_up:
            uploaded = st.file_uploader(
                "WhatsApp export (.txt, or .zip exported 'Without media')",
                type=["txt", "zip"],
                key=f"wa_upload_{_ss()['wa_uploader_gen']}",
                max_upload_size=config.MAX_UPLOAD_MB,
                help=f"Maximum {config.MAX_UPLOAD_MB} MB. In WhatsApp: open the chat → ⋮ / contact name → "
                "More → Export chat → Without media.",
            )
        with col_side:
            st.markdown("**No export at hand?**")
            st.button(
                "Try the synthetic sample chat",
                on_click=load_sample,
                help="A small invented group chat. Contains no real people or conversations.",
            )
            has_data = uploaded is not None or _ss().get("wa_sample_loaded")
            st.button(
                "Clear chat and results",
                on_click=clear_session,
                disabled=not has_data,
                type="secondary",
                help="Removes the chat, all derived results and AI summaries from this session.",
            )
    if _ss().get("wa_cleared") and uploaded is None and not _ss().get("wa_sample_loaded"):
        st.success("Cleared. The chat and all results were removed from this session.")
    if uploaded is not None:
        raw = uploaded.getvalue()
        if len(raw) > config.MAX_UPLOAD_BYTES:
            st.error(f"The file is larger than the {config.MAX_UPLOAD_MB} MB limit.")
            return None
        _ss()["wa_sample_loaded"] = False
        _ss()["wa_cleared"] = False
        source = (hashlib.sha256(raw).hexdigest(), raw, "uploaded file")
    elif _ss().get("wa_sample_loaded"):
        raw = SAMPLE_CHAT.encode("utf-8")
        source = (hashlib.sha256(raw).hexdigest(), raw, "synthetic sample chat")
    else:
        return None
    _drop_stale(source[0])
    return source


def parse_step(sha: str, raw: bytes, label: str) -> parsing.ParseResult | None:
    st.subheader("2 · Review parsing")
    order = _ss().get("wa_date_order", "auto")
    key = ("wa_parsed", sha, order)
    if _ss().get("wa_parsed_key") != key:
        try:
            with st.spinner("Reading the chat…"):
                text, read_warnings = parsing.read_upload(
                    raw, "upload.zip" if raw[:4] == b"PK\x03\x04" else "upload.txt"
                )
                result = parsing.parse_chat(text, date_order=order)
                result.report.warnings[:0] = read_warnings
        except parsing.ParseError as exc:
            st.error(str(exc))
            _date_order_control(None)
            return None
        _ss()["wa_parsed_key"] = key
        _ss()["wa_parsed_result"] = result
    result: parsing.ParseResult = _ss()["wa_parsed_result"]
    rep = result.report
    needs_attention = bool(rep.warnings) or rep.date_order_ambiguous
    with st.container(border=True):
        st.caption(f"Source: {label}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records parsed", f"{rep.messages_parsed:,}")
        c2.metric("Participants", f"{rep.participants:,}")
        c3.metric("First message", f"{rep.first_timestamp:%Y-%m-%d}")
        c4.metric("Last message", f"{rep.last_timestamp:%Y-%m-%d}")
        st.markdown(
            f"**Detected format:** {md_escape(rep.detected_format)}  \n"
            f"**Date order:** {md_escape(rep.date_order)} — {md_escape(rep.date_order_source)}"
        )
        tc = rep.type_counts
        st.markdown(
            f"Text: **{tc['text']:,}** · Media placeholders: **{tc['media']:,}** · "
            f"Deleted markers: **{tc['deleted']:,}** · System events: **{tc['system']:,}** · "
            f"Skipped: **{rep.skipped_records:,}**"
        )
        if rep.date_order_ambiguous:
            st.warning(
                "Day and month could not be told apart with certainty. Check the date range above and "
                "choose the date order explicitly below if it looks wrong."
            )
        for w in rep.warnings:
            if "ambiguous" not in w:
                st.info(w)
        with st.expander("Parsing options and details", expanded=needs_attention):
            _date_order_control(rep)
            if rep.skipped_details:
                st.markdown("Skipped records (line numbers only):")
                st.dataframe(
                    pd.DataFrame([{"line": d.line_no, "reason": d.reason} for d in rep.skipped_details]),
                    hide_index=True,
                    height=min(240, 40 + 35 * len(rep.skipped_details)),
                )
            st.caption(
                "Times are shown exactly as exported (the phone's local time). The export contains no "
                "timezone, so none is assumed."
            )
    return result


def _date_order_control(report: parsing.ParseReport | None) -> None:
    st.radio(
        "Date order",
        options=list(parsing.DATE_ORDER_LABELS),
        format_func=parsing.DATE_ORDER_LABELS.get,
        key="wa_date_order",
        horizontal=True,
        help="WhatsApp writes dates in the phone's regional format. Choose explicitly if detection is ambiguous.",
    )


def filter_step(df: pd.DataFrame, sha: str) -> tuple[analytics.Filters, int] | None:
    st.subheader("3 · Choose participants and dates")
    people = memo(("participants", sha, _ss().get("wa_date_order")), lambda: analytics.participants(df))
    if not people:
        st.info("This export contains no participant messages to analyse.")
        return None
    first = df["timestamp"].min().date()
    last = df["timestamp"].max().date()
    suffix = f"{sha[:12]}_{_ss().get('wa_date_order', 'auto')}"
    with st.container(border=True):
        col_p, col_d, col_i = st.columns([3, 2, 1.4])
        with col_p:
            chosen = st.multiselect(
                "Participants",
                options=people,
                default=people,
                key=f"wa_f_people_{suffix}",
                help="All metrics, charts, downloads and AI inputs use this selection.",
            )
        with col_d:
            picked = st.date_input(
                "Date range",
                value=(first, last),
                min_value=first,
                max_value=last,
                key=f"wa_f_dates_{suffix}",
                format="YYYY-MM-DD",
            )
        with col_i:
            idle = st.selectbox(
                "Session idle gap",
                options=list(IDLE_OPTIONS),
                index=1,
                format_func=IDLE_OPTIONS.get,
                key=f"wa_f_idle_{suffix}",
                help="A new conversation session starts after a gap longer than this.",
            )
    if not chosen:
        st.info("Select at least one participant.")
        return None
    if isinstance(picked, (tuple, list)):
        start, end = (picked[0], picked[1]) if len(picked) == 2 else (picked[0], picked[0])
    else:
        start = end = picked
    participants_filter = None if set(chosen) == set(people) else tuple(sorted(chosen))
    return analytics.Filters(participants=participants_filter, start=start, end=end), int(idle)


def describe_filters(filters: analytics.Filters) -> str:
    who = "all participants" if filters.participants is None else f"{len(filters.participants)} selected participant(s)"
    return f"{who}; {filters.start:%Y-%m-%d} to {filters.end:%Y-%m-%d}"


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def overview_tab(base_key: tuple, filtered: pd.DataFrame) -> None:
    ov: analytics.Overview = memo(base_key + ("overview",), lambda: analytics.overview(filtered))
    if ov.messages == 0:
        st.info("No messages match the current selection.")
        return
    r1 = st.columns(4)
    r1[0].metric(
        "Messages", f"{ov.messages:,}", help="Text + media placeholders + deleted markers. System events excluded."
    )
    r1[1].metric(
        "Words",
        f"{ov.words:,}",
        help="Whitespace-separated tokens containing a letter or digit, in text messages; links excluded.",
    )
    r1[2].metric("Media placeholders", f"{ov.media_messages:,}")
    r1[3].metric("Links", f"{ov.links:,}")
    r2 = st.columns(4)
    r2[0].metric("Active days", f"{ov.active_days:,}", help="Calendar dates with at least one message.")
    r2[1].metric(
        "Calendar span", f"{ov.calendar_days:,} days", help="First to last message in the selection, inclusive."
    )
    r2[2].metric("Avg / active day", f"{ov.avg_per_active_day:,.1f}", help="Messages ÷ active days.")
    r2[3].metric(
        "Avg / calendar day",
        f"{ov.avg_per_calendar_day:,.2f}",
        help="Messages ÷ calendar span (days with no messages count as 0).",
    )
    st.caption(
        f"Also in this selection: {ov.text_messages:,} text messages, {ov.deleted_messages:,} deleted-message markers, "
        f"{ov.emoji:,} emoji, {ov.system_events:,} system events (not counted as messages)."
    )
    parts = memo(base_key + ("participants_table",), lambda: analytics.participant_summary(filtered))
    left, right = st.columns([3, 2])
    with left:
        top = parts.head(config.CHART_TOP_N)
        title = "Messages per participant" + (f" (top {config.CHART_TOP_N})" if len(parts) > config.CHART_TOP_N else "")
        show_chart(
            hbar(top["participant"].tolist(), top["messages"].tolist(), title, "Messages"),
            height=max(220, 40 + 28 * len(top)),
        )
    with right:
        st.dataframe(
            parts.head(config.TABLE_ROW_LIMIT),
            hide_index=True,
            height=min(420, 40 + 35 * min(len(parts), 11)),
            column_config={"share_pct": st.column_config.NumberColumn("share %", format="%.1f")},
        )


def activity_tab(
    base_key: tuple, full_df: pd.DataFrame, filtered: pd.DataFrame, filters: analytics.Filters, idle: int
) -> None:
    ov = memo(base_key + ("overview",), lambda: analytics.overview(filtered))
    if ov.messages == 0:
        st.info("No messages match the current selection.")
        return
    with safe_section("timeline"):
        grain = st.radio(
            "Timeline granularity",
            ["Daily", "Weekly", "Monthly"],
            index=0 if ov.calendar_days <= 120 else (1 if ov.calendar_days <= 730 else 2),
            horizontal=True,
            key="wa_grain",
        )
        if grain == "Daily":
            series = memo(base_key + ("daily",), lambda: analytics.daily_counts(filtered))
            x, label = series["date"], "Day"
        elif grain == "Weekly":
            series = memo(base_key + ("weekly",), lambda: analytics.weekly_counts(filtered))
            x, label = series["week_start"], "Week starting"
        else:
            series = memo(base_key + ("monthly",), lambda: analytics.monthly_counts(filtered))
            x, label = series["month"], "Month"
        fig = go.Figure(
            go.Scatter(
                x=x,
                y=series["messages"],
                mode="lines",
                line=dict(color=ACCENT, width=2),
                hovertemplate=f"{label} %{{x|%Y-%m-%d}}<br>Messages: %{{y:,}}<extra></extra>",
            )
        )
        fig.update_layout(title=f"{grain} messages (periods with no messages shown as 0)", hovermode="x unified")
        show_chart(fig, height=320)

    with safe_section("heatmap"):
        matrix = memo(base_key + ("heatmap",), lambda: analytics.weekday_hour_matrix(filtered))
        fig = go.Figure(
            go.Heatmap(
                z=matrix.to_numpy(),
                x=[f"{h:02d}" for h in matrix.columns],
                y=list(matrix.index),
                colorscale=HEATMAP_SCALE,
                colorbar=dict(title="Messages"),
                xgap=2,
                ygap=2,
                hovertemplate="%{y} %{x}:00<br>Messages: %{z:,}<extra></extra>",
            )
        )
        fig.update_layout(
            title="When messages are sent: weekday × hour (export's local time)",
            yaxis=dict(autorange="reversed"),
            xaxis_title="Hour of day",
        )
        show_chart(fig, height=330)

    with safe_section("conversation timing"):
        st.markdown("#### Conversation sessions")
        sess, starters = memo(base_key + ("sessions", idle), lambda: analytics.sessions(full_df, filters, idle))
        st.caption(
            f"A session is a run of messages with no gap longer than {IDLE_OPTIONS[idle]}. The *session starter* "
            "is whoever sent the first message after such a gap — this describes timing only, not effort or intent."
        )
        if sess.empty:
            st.info("No sessions start inside the selected range.")
        else:
            c = st.columns(3)
            c[0].metric("Sessions", f"{len(sess):,}")
            c[1].metric("Median messages / session", f"{sess['messages'].median():,.0f}")
            c[2].metric("Median session length", f"{sess['duration_min'].median():,.0f} min")
            st.dataframe(starters, hide_index=True, height=min(300, 40 + 35 * len(starters)))

        st.markdown("#### Response-gap estimates")
        gaps = memo(base_key + ("gaps", idle), lambda: analytics.response_gaps(full_df, filters, idle))
        st.caption(
            "Estimated from the complete conversation: the time before a message whose sender differs from the "
            "previous message's sender, within a session. In group chats the previous message is not necessarily "
            "what was being answered, so treat these as estimates. Follow-ups are consecutive messages by the same sender."
        )
        st.dataframe(gaps, hide_index=True, height=min(300, 40 + 35 * max(len(gaps), 1)))

        st.markdown("#### Longest quiet periods")
        silences = memo(base_key + ("silence",), lambda: analytics.silence_gaps(full_df, filters, min_hours=24))
        st.caption(
            "Gaps of 24 hours or more between consecutive messages in the whole conversation (all participants), "
            "ending inside the selected dates. They show when the chat was quiet — not who was responsible."
        )
        if silences.empty:
            st.info("No gaps of 24 hours or more in this range.")
        else:
            st.dataframe(
                silences,
                hide_index=True,
                column_config={
                    "gap_start": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                    "gap_end": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                },
            )


def words_tab(base_key: tuple, filtered: pd.DataFrame) -> None:
    freqs = memo(base_key + ("freqs",), lambda: analytics.word_frequencies(filtered))
    words = pd.DataFrame(list(freqs.items())[:20], columns=["word", "count"])  # same counts as the word cloud
    emo = memo(base_key + ("emoji",), lambda: analytics.top_emoji(filtered, n=15))
    left, right = st.columns(2)
    with left, safe_section("words"):
        if words.empty:
            st.info("No words to show for this selection.")
        else:
            show_chart(
                hbar(words["word"].tolist(), words["count"].tolist(), "Most frequent words", "Count"),
                height=max(240, 40 + 24 * len(words)),
            )
            st.caption("Common English stop words are excluded. Other languages are counted but not filtered.")
    with right, safe_section("emoji"):
        if emo.empty:
            st.info("No emoji in this selection.")
        else:
            show_chart(
                hbar(emo["emoji"].tolist(), emo["count"].tolist(), "Most used emoji", "Count"),
                height=max(240, 40 + 26 * len(emo)),
            )
            st.caption("Each full emoji sequence counts once (e.g. 👍🏽, 🇮🇳, 👨‍👩‍👧).")
    with safe_section("word cloud"):
        if freqs and st.toggle("Show word cloud", key="wa_show_wc"):
            img = memo(base_key + ("wordcloud",), lambda: _word_cloud_image(freqs))
            if img is None:
                st.info("Word clouds are unavailable (the optional 'wordcloud' package is not installed).")
            else:
                st.image(
                    img,
                    caption="Word cloud of frequent words (size ∝ frequency). Non-Latin scripts may not render "
                    "with the default font.",
                    width="stretch",
                )


def _word_cloud_image(freqs: dict[str, int]):
    try:
        from wordcloud import WordCloud
    except ImportError:
        return None
    wc = WordCloud(
        width=900, height=380, background_color="white", colormap="Greens", max_words=150, prefer_horizontal=0.95
    )
    return wc.generate_from_frequencies(freqs).to_image()


@st.cache_resource(show_spinner=False, max_entries=1)
def _load_sentiment_model() -> sentiment.HFSentimentModel:
    # Shared across sessions on purpose: immutable public model weights only.
    return sentiment.HFSentimentModel()


def sentiment_tab(base_key: tuple, filtered: pd.DataFrame) -> None:
    st.markdown(
        f"Optional message-level sentiment using the public model `{sentiment.MODEL_ID}` (negative / neutral / positive). "
        "It was fine-tuned on tweets in Arabic, English, French, German, Hindi, Italian, Spanish and Portuguese and has "
        "**not** been evaluated on WhatsApp chats. Expect errors with informal language, sarcasm, romanised Hindi / "
        "Hinglish, Gujarati and mixed-language messages. Use it as a rough, exploratory signal only."
    )
    if not sentiment.dependencies_available():
        st.info(
            "Sentiment analysis is not installed on this deployment. To enable it locally run "
            "`pip install -r requirements-ml.txt` (CPU-only PyTorch, about 1 GB model download on first use)."
        )
        return
    n_text = int((filtered["message_type"] == "text").sum())
    if n_text == 0:
        st.info("No text messages in the current selection.")
        return
    cap = config.SENTIMENT_MAX_MESSAGES
    key = base_key + ("sentiment", cap)
    runs: OrderedDict = _ss()["wa_sentiment_runs"]
    st.caption(
        f"{n_text:,} text messages selected"
        + (f"; the most recent {cap:,} will be analysed." if n_text > cap else ".")
        + " Runs on this server's CPU; the model is downloaded from Hugging Face on first use."
    )
    if st.button("Run sentiment analysis", key="wa_run_sentiment", disabled=key in runs):
        try:
            with st.spinner("Loading the sentiment model…"):
                model = _load_sentiment_model()
            bar = st.progress(0.0, text="Classifying messages…")
            run = sentiment.analyse(filtered, model, progress=lambda f: bar.progress(f, text="Classifying messages…"))
            bar.empty()
            _lru_put(runs, key, run, size=3)
        except sentiment.SentimentError as exc:
            st.error(str(exc))
            return
    run = runs.get(key)
    if run is None:
        st.info("Sentiment has not been run for this selection yet.")
        return
    if run.failed:
        st.warning(f"{run.failed:,} message(s) could not be classified and are excluded.")
    st.caption(f"Classified {run.analysed:,} messages; {run.not_analysed_limit:,} older messages not analysed (limit).")
    with safe_section("sentiment"):
        dist = sentiment.label_distribution(filtered, run)
        participants_order = dist.groupby("participant")["messages"].sum().sort_values(ascending=False).index.tolist()
        participants_order = participants_order[: config.CHART_TOP_N]
        fig = go.Figure()
        for label in sentiment.CLASSES:
            sub = dist[dist["label"] == label].set_index("participant").reindex(participants_order)
            fig.add_bar(
                name=label,
                y=[safe_label(p) for p in participants_order],
                x=sub["share_pct"].fillna(0),
                orientation="h",
                marker=dict(color=SENTIMENT_COLORS[label]),
                customdata=sub["messages"].fillna(0),
                hovertemplate="%{y}<br>" + label + ": %{x:.1f}% (%{customdata:,.0f} messages)<extra></extra>",
            )
        fig.update_layout(
            barmode="stack",
            title="Predicted label share per participant (%)",
            xaxis=dict(range=[0, 100]),
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", y=-0.15),
        )
        show_chart(fig, height=max(260, 80 + 30 * len(participants_order)))

        monthly = sentiment.monthly_summary(filtered, run)
        if len(monthly) >= 2:
            fig = go.Figure()
            for label in sentiment.CLASSES:
                fig.add_bar(
                    name=label,
                    x=monthly["month"],
                    y=monthly[f"p_{label}"],
                    marker=dict(color=SENTIMENT_COLORS[label]),
                    customdata=monthly["n"],
                    hovertemplate="%{x|%Y-%m}<br>" + label + ": %{y:.2f} (n=%{customdata:,})<extra></extra>",
                )
            fig.update_layout(
                barmode="stack",
                title="Average class probability per month",
                yaxis=dict(range=[0, 1]),
                legend=dict(orientation="h", y=-0.2),
            )
            show_chart(fig, height=320)
        st.caption(
            "Shares count each message's most likely label. The model's confidence in a label is *not* a "
            "positive/negative score. Where a single number is shown, the polarity index = P(positive) − P(negative), "
            "ranging from −1 to 1; it is an uncalibrated summary of the class probabilities."
        )
        with st.expander("Inspect a sample of classified messages (visible only to you)"):
            ok = run.results[run.results["status"] == "ok"].tail(100)
            view = pd.DataFrame(
                {
                    "time": filtered.loc[ok.index, "timestamp"],
                    "participant": filtered.loc[ok.index, "sender"],
                    "message": filtered.loc[ok.index, "text"].str.slice(0, 200),
                    "label": ok["label"],
                    "confidence": ok["confidence"].round(2),
                    "polarity_index": ok["polarity"].round(2),
                }
            )
            st.dataframe(view, hide_index=True, height=300)


def ai_tab(base_key: tuple, filtered: pd.DataFrame, filters: analytics.Filters) -> None:
    settings = config.load_ai_settings()
    st.markdown(
        "Generate a factual summary of the **selected** messages: topics, explicitly stated decisions, open questions "
        "and action items. Nothing is sent unless you review the content below, confirm, and click the button."
    )
    if not settings.usable:
        st.info("AI summaries are disabled on this deployment.")
        return
    with st.container(border=True):
        st.markdown(
            f"**Provider:** {ai_summary.PROVIDER_NAME} (`{settings.base_url}`), a third-party service. The selected "
            "content leaves this server and is handled under the provider's terms and privacy policy. Check whether the "
            "account used has data retention and training disabled."
        )
        c1, c2, c3 = st.columns([2, 1, 1])
        model = c1.selectbox(
            "Model",
            settings.models,
            key="wa_ai_model",
            help="Configure with WA_TOGETHER_MODELS. Verify IDs against Together's current model list.",
        )
        pseudo = c2.checkbox(
            "Use pseudonyms",
            value=True,
            key="wa_ai_pseudo",
            help="Replace participant names (also inside messages) with 'Participant A', 'B', …",
        )
        redact = c3.checkbox(
            "Mask identifiers",
            value=True,
            key="wa_ai_redact",
            help="Mask e-mail addresses, phone numbers, links and @mentions.",
        )
        st.caption(
            "Pseudonyms and masking reduce exposure but do not guarantee anonymity: message content can still "
            "identify people."
        )

    transcript = memo(
        base_key + ("transcript", pseudo, redact),
        lambda: ai_summary.build_transcript(filtered, pseudonymize=pseudo, redact=redact),
    )
    plan = memo(base_key + ("plan", pseudo, redact), lambda: ai_summary.plan_chunks(transcript.lines))
    if not plan.chunks:
        st.info("There are no text messages in the current selection to summarise.")
        return
    key = ai_summary.cache_key(model, plan.chunks, config.AI_MAX_OUTPUT_TOKENS)
    n_requests = ai_summary.requests_needed(len(plan.chunks))
    st.markdown(
        f"**Will send:** {plan.lines_included:,} messages ({describe_filters(filters)})"
        + (
            f"; the {plan.lines_omitted:,} oldest selected messages are left out to stay within limits"
            if plan.lines_omitted
            else ""
        )
        + f". About {plan.estimated_input_tokens:,} input tokens (estimate) in {n_requests} request(s); "
        f"at most {config.AI_MAX_OUTPUT_TOKENS:,} output tokens per request."
    )
    with st.expander("Review exactly what will be sent"):
        st.code("\n\n--- next request ---\n\n".join(plan.chunks), language=None, wrap_lines=True, height=300)
        if transcript.pseudonyms:
            st.dataframe(
                pd.DataFrame({"name": list(transcript.pseudonyms), "sent as": list(transcript.pseudonyms.values())}),
                hide_index=True,
            )

    results: OrderedDict = _ss()["wa_ai_results"]
    if key in results:
        _show_summary(results[key], cached=True)
        return

    api_key = settings.shared_key
    if api_key is None and settings.allow_user_key:
        api_key = st.text_input(
            "Your Together AI API key",
            type="password",
            key="wa_ai_user_key",
            help="Used only for this session's requests; kept in memory, never stored or logged.",
        )
    used = _ss()["wa_ai_requests"]
    remaining = settings.max_requests_per_session - used
    if _ss().get("wa_ai_consent_for") != key:
        # Consent applies to one exact payload: any change to content or model requires it again.
        _ss()["wa_ai_consent"] = False
        _ss()["wa_ai_consent_for"] = key
    consent = st.checkbox(
        "I have reviewed the content above, I am allowed to share it, and I want to send it to "
        f"{ai_summary.PROVIDER_NAME} for summarisation.",
        key="wa_ai_consent",
    )
    blocked_reason = None
    if not api_key:
        blocked_reason = "Enter an API key to enable this action."
    elif remaining < n_requests:
        blocked_reason = "The AI request limit for this session has been reached."
    if blocked_reason:
        st.caption(blocked_reason)
    clicked = st.button(
        "Generate AI summary", type="primary", disabled=not consent or blocked_reason is not None, key="wa_ai_generate"
    )
    if clicked and consent and api_key and remaining >= n_requests:
        _run_summary(api_key, settings, model, plan, key)


def _run_summary(api_key: str, settings: config.AISettings, model: str, plan: ai_summary.ChunkPlan, key: str) -> None:
    def count() -> None:
        _ss()["wa_ai_requests"] += 1

    def budget() -> bool:
        return _ss()["wa_ai_requests"] < settings.max_requests_per_session

    status = st.status("Contacting the AI provider…", expanded=True)
    try:
        client = ai_summary.TogetherClient(api_key, base_url=settings.base_url)
        result = ai_summary.summarize(
            client, model, plan, budget=budget, on_request=count, on_progress=lambda msg: status.update(label=msg)
        )
    except ai_summary.ProviderError as exc:
        status.update(label="AI summary failed", state="error")
        st.error(exc.user_message)
        if exc.partial_text:
            with st.expander("Incomplete output received before the error"):
                st.markdown(ai_summary.render_safe_markdown(exc.partial_text))
        return
    status.update(label="Summary ready", state="complete", expanded=False)
    _lru_put(_ss()["wa_ai_results"], key, result, size=5)
    _show_summary(result, cached=False)


def _show_summary(result: ai_summary.SummaryResult, cached: bool) -> None:
    with st.container(border=True):
        st.markdown("**AI-generated summary** — may be incomplete or wrong; check important details against the chat.")
        st.markdown(ai_summary.render_safe_markdown(result.text))
        st.caption(
            f"Model {result.model} · prompt {result.prompt_version} · {result.requests_made} request(s) · "
            f"{result.lines_included:,} messages included"
            + (
                " · Showing the saved result for this exact content and model; no new request was made."
                if cached
                else ""
            )
        )
        st.download_button(
            "Download summary (.txt)",
            ai_summary.sanitize_output(result.text).encode("utf-8"),
            file_name="ai-summary.txt",
            mime="text/plain",
            on_click="ignore",
            key="wa_ai_dl",
        )


def export_tab(
    base_key: tuple, report: parsing.ParseReport, filtered: pd.DataFrame, filters: analytics.Filters
) -> None:
    st.markdown(
        "Download aggregate statistics for the current selection. Downloads contain **no message text**. "
        "Text fields are protected against spreadsheet formula injection."
    )
    pseudo = st.checkbox("Replace participant names with pseudonyms in downloads", key="wa_export_pseudo")

    def build() -> tuple[bytes, str]:
        ov = analytics.overview(filtered)
        parts = analytics.participant_summary(filtered)
        if pseudo:
            mapping = export.pseudonym_map(parts["participant"].tolist())
            parts = parts.assign(participant=parts["participant"].map(mapping))
        report_md = export.build_report(report, ov, parts, describe_filters(filters))
        tables = {
            "overview": export.overview_table(ov),
            "participants": parts,
            "daily_messages": analytics.daily_counts(filtered),
            "monthly_messages": analytics.monthly_counts(filtered),
            "weekday_hour": analytics.weekday_hour_matrix(filtered).reset_index(names="weekday"),
            "top_words": analytics.top_words(filtered, n=50),
            "top_emoji": analytics.top_emoji(filtered, n=30),
        }
        return export.build_zip(tables, report_md), report_md

    data, report_md = memo(base_key + ("export", pseudo), build)
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download statistics (.zip of CSV files + report)",
        data,
        file_name="chat-statistics.zip",
        mime="application/zip",
        on_click="ignore",
        key="wa_dl_zip",
    )
    c2.download_button(
        "Download report (.md)",
        report_md.encode("utf-8"),
        file_name="chat-report.md",
        mime="text/markdown",
        on_click="ignore",
        key="wa_dl_report",
    )
    if not pseudo:
        st.caption("Participant names are included. Tick the box above before sharing downloads with others.")


def footer() -> None:
    st.divider()
    st.caption(
        "WhatsApp Chat Analyzer · created by Rajeeb Lochan Behera · "
        "[GitHub](https://github.com/RajeebLochan) · [LinkedIn](https://linkedin.com/in/rajeeb-lochan) · "
        "Not affiliated with WhatsApp or Meta."
    )


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "- Processing happens on the server running this app, in memory, per session.\n"
            "- No chat is saved to disk or a database by the app.\n"
            "- AI summaries are off until you opt in; they send the selected text to the provider.\n"
            "- The optional sentiment model is downloaded from Hugging Face on first use."
        )
        st.caption(f"Limits: {config.MAX_UPLOAD_MB} MB per file, {config.MAX_MESSAGES:,} messages.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="WhatsApp Chat Analyzer", page_icon="💬", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)  # static, no user content
    init_state()
    sidebar()
    header()
    source = upload_step()
    if source is None:
        st.info("Upload an export or try the synthetic sample to begin.")
        footer()
        return
    sha, raw, label = source
    result = parse_step(sha, raw, label)
    if result is None:
        footer()
        return
    df = result.df
    picked = filter_step(df, sha)
    if picked is None:
        footer()
        return
    filters, idle = picked
    order = _ss().get("wa_date_order", "auto")
    base_key = (sha, order, filters.key())
    filtered = memo(base_key + ("filtered",), lambda: analytics.apply_filters(df, filters))

    st.subheader("4 · Explore")
    tabs = st.tabs(
        ["Overview", "Activity", "Words & emoji", "Sentiment (optional)", "AI summary (optional)", "Download"]
    )
    with tabs[0], safe_section("overview"):
        overview_tab(base_key, filtered)
    with tabs[1], safe_section("activity"):
        activity_tab(base_key, df, filtered, filters, idle)
    with tabs[2], safe_section("words and emoji"):
        words_tab(base_key, filtered)
    with tabs[3], safe_section("sentiment"):
        sentiment_tab(base_key, filtered)
    with tabs[4], safe_section("AI summary"):
        ai_tab(base_key, filtered, filters)
    with tabs[5], safe_section("download"):
        export_tab(base_key, result.report, filtered, filters)
    footer()
