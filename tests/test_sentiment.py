"""Sentiment logic tests with a fake model (no torch / downloads needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wa_analyzer import sentiment as se
from wa_analyzer.parsing import parse_chat

CHAT = (
    "01/03/2024, 09:00 - A: I love this 😍\n"
    "01/03/2024, 09:01 - B: <Media omitted>\n"
    "01/03/2024, 09:02 - B: this is terrible\n"
    "01/03/2024, 09:03 - A: FAIL please\n"
    "01/03/2024, 09:04 - Messages and calls are end-to-end encrypted.\n"
    "01/03/2024, 09:05 - B: meh @Alice see https://x.org\n"
)


class FakeModel:
    """Label order deliberately differs from model-card order to test mapping."""

    def __init__(self, id2label=None):
        self.label_map = se.normalize_label_map(id2label or {0: "Positive", 1: "Negative", 2: "Neutral"})
        self.seen: list[list[str]] = []

    def predict_proba(self, texts):
        self.seen.append(list(texts))
        if any("FAIL" in t for t in texts):
            raise RuntimeError("boom")
        rows = []
        for t in texts:
            if "love" in t:
                rows.append([0.90, 0.02, 0.08])  # positive
            elif "terrible" in t:
                rows.append([0.01, 0.95, 0.04])  # confident NEGATIVE
            else:
                rows.append([0.10, 0.20, 0.70])  # neutral
        return np.array(rows)


@pytest.fixture()
def df():
    return parse_chat(CHAT).df


def test_normalize_label_map_variants():
    assert se.normalize_label_map({"0": "negative", "1": "neutral", "2": "positive"}) == {
        0: "negative",
        1: "neutral",
        2: "positive",
    }
    assert se.normalize_label_map({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}) == {
        0: "negative",
        1: "neutral",
        2: "positive",
    }
    with pytest.raises(se.SentimentError):
        se.normalize_label_map({0: "joy", 1: "anger"})


def test_preprocess_follows_model_card_and_keeps_emoji():
    assert se.preprocess("hi @Alice see https://x.org 😍") == "hi @user see http 😍"
    assert se.preprocess(None) == ""


def test_confident_negative_is_negative_not_positive(df):
    run = se.analyse(df, FakeModel(), batch_size=8)
    r = run.results
    neg_row = df.index[df["text"] == "this is terrible"][0]
    assert r.loc[neg_row, "label"] == "negative"
    assert r.loc[neg_row, "confidence"] == pytest.approx(0.95)
    assert r.loc[neg_row, "polarity"] == pytest.approx(0.01 - 0.95)  # strongly negative index
    pos_row = df.index[df["text"].str.startswith("I love")][0]
    assert r.loc[pos_row, "label"] == "positive"
    assert r.loc[pos_row, "p_positive"] == pytest.approx(0.90)


def test_row_alignment_with_skips_and_failures(df):
    model = FakeModel()
    run = se.analyse(df, model, batch_size=8)
    r = run.results
    assert list(r.index) == list(df.index)
    by_text = dict(zip(df["text"], r["status"], strict=True))
    assert by_text["<Media omitted>"] == "skipped"
    assert by_text["Messages and calls are end-to-end encrypted."] == "skipped"
    assert by_text["FAIL please"] == "error"
    assert by_text["this is terrible"] == "ok"  # neighbours of a failed item still align
    assert run.analysed == 3 and run.failed == 1
    # batch failed once, then items were retried individually
    assert len(model.seen[0]) == 4 and all(len(b) == 1 for b in model.seen[1:])
    # model input used card preprocessing
    assert "meh @user see http" in model.seen[0]


def test_message_cap_keeps_most_recent(df):
    run = se.analyse(df, FakeModel(), max_messages=2)
    statuses = run.results["status"].tolist()
    assert statuses.count("not analysed (limit)") == 2
    assert run.not_analysed_limit == 2
    assert run.results.loc[df.index[-1], "status"] == "ok"


def test_aggregations(df):
    run = se.analyse(df, FakeModel())
    dist = se.label_distribution(df, run)
    b = dist[dist["participant"] == "B"].set_index("label")
    assert b.loc["negative", "messages"] == 1 and b.loc["neutral", "messages"] == 1
    assert b["share_pct"].sum() == pytest.approx(100.0)
    monthly = se.monthly_summary(df, run)
    assert monthly["n"].tolist() == [3]
    assert -1 <= monthly.loc[0, "polarity"] <= 1


def test_empty_selection():
    empty = parse_chat(CHAT).df.iloc[0:0]
    run = se.analyse(empty, FakeModel())
    assert run.analysed == 0
    assert se.label_distribution(empty, run).empty
    assert se.monthly_summary(empty, run).empty


def test_dependencies_check_is_cheap():
    assert isinstance(se.dependencies_available(), bool)


def test_results_dtype_numeric(df):
    run = se.analyse(df, FakeModel())
    assert pd.api.types.is_float_dtype(run.results["confidence"])
