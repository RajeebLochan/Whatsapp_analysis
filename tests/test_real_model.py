"""Optional checks against the real Hugging Face sentiment model.

Skipped by default. Run with ``WA_RUN_REAL_MODEL=1`` after installing
``requirements-ml.txt`` (needs network access to huggingface.co).

What this validates: the model downloads, its label mapping normalises to
negative/neutral/positive, inference runs on CPU in batches, and two
unambiguous English sentences land on the expected side. It does NOT
measure accuracy on WhatsApp chats or any language — no evaluation set exists
in this project.
"""

from __future__ import annotations

import os

import pytest

from wa_analyzer import sentiment
from wa_analyzer.parsing import parse_chat

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(os.environ.get("WA_RUN_REAL_MODEL") != "1", reason="set WA_RUN_REAL_MODEL=1 to run"),
    pytest.mark.skipif(not sentiment.dependencies_available(), reason="requirements-ml.txt not installed"),
]


def test_real_model_labels_and_sanity():
    model = sentiment.HFSentimentModel()
    assert sorted(model.label_map.values()) == sorted(sentiment.CLASSES)
    df = parse_chat(
        "01/03/2024, 09:00 - A: I love this, thank you so much! 😍\n"
        "01/03/2024, 09:01 - B: This is terrible, I hate it.\n"
        "01/03/2024, 09:02 - A: <Media omitted>\n"
    ).df
    run = sentiment.analyse(df, model, batch_size=2)
    labels = run.results["label"].tolist()
    assert labels[0] == "positive"
    assert labels[1] == "negative"
    assert run.results["status"].tolist() == ["ok", "ok", "skipped"]
    probs = run.results[["p_negative", "p_neutral", "p_positive"]].iloc[:2].sum(axis=1)
    assert probs.round(4).tolist() == [1.0, 1.0]
