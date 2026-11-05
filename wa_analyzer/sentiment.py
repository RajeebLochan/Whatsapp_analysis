"""Optional message-level sentiment classification.

Model: ``cardiffnlp/twitter-xlm-roberta-base-sentiment`` (XLM-RoBERTa base
trained on tweets, fine-tuned for 3-class sentiment). Its card says it was
fine-tuned on Arabic, English, French, German, Hindi, Italian, Spanish and
Portuguese tweets. It was **not** evaluated by this project on WhatsApp
messages, and accuracy is unknown for informal chat, sarcasm, romanised
Hindi/Hinglish, Gujarati or code-mixed text. Treat results as rough,
exploratory signals.

Design
------
- ``torch``/``transformers`` are imported lazily, only when the user asks for
  sentiment; the rest of the app works without them (``requirements-ml.txt``).
- The model's own ``id2label`` mapping is read and normalised to
  ``negative``/``neutral``/``positive``; ``LABEL_n`` names fall back to the
  model-card order (0 negative, 1 neutral, 2 positive).
- Class *probabilities* are kept per message. The predicted label's
  probability is its *confidence*, which is not a polarity score.
- A derived *polarity index* = P(positive) − P(negative), in [−1, 1]. It is an
  uncalibrated summary of the class distribution, not a measured quantity.
- Inputs follow the model card's preprocessing (``@mention`` → ``@user``,
  links → ``http``); emoji are kept because the model was trained with them.
- Inference is batched with bounded text length; a failing batch is retried
  item by item and failures are marked per row without shifting alignment.
"""

from __future__ import annotations

import importlib.util
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
CLASSES = ("negative", "neutral", "positive")
_INDEX_FALLBACK = {0: "negative", 1: "neutral", 2: "positive"}  # documented on the model card
MAX_CHARS = 1_000  # bound tokenizer work per message before truncation to tokens
RESULT_COLUMNS = ["status", "label", "confidence", "p_negative", "p_neutral", "p_positive", "polarity"]


class SentimentError(RuntimeError):
    """User-safe sentiment failure."""


class SentimentModel(Protocol):
    label_map: dict[int, str]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...


def dependencies_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


def normalize_label_map(id2label: Mapping[object, object]) -> dict[int, str]:
    out: dict[int, str] = {}
    for key, value in id2label.items():
        idx = int(key)  # config keys may be strings
        name = str(value).strip().lower()
        m = re.fullmatch(r"label_(\d+)", name)
        if m:
            name = _INDEX_FALLBACK.get(int(m.group(1)), name)
        if name not in CLASSES:
            raise SentimentError(f"Unrecognised sentiment label in model configuration: {value!r}")
        out[idx] = name
    if sorted(out.values()) != sorted(CLASSES):
        raise SentimentError("The model does not expose exactly negative/neutral/positive classes.")
    return out


def preprocess(text: str) -> str:
    """Model-card preprocessing: mask user handles and links; keep emoji."""
    if not isinstance(text, str):
        return ""
    tokens = []
    for tok in text[:MAX_CHARS].split():
        if tok.startswith("@") and len(tok) > 1:
            tok = "@user"
        elif tok.startswith("http") or tok.startswith("www."):
            tok = "http"
        tokens.append(tok)
    return " ".join(tokens)


class HFSentimentModel:
    """Hugging Face implementation. Construct only after ``dependencies_available()``."""

    def __init__(self, model_id: str = MODEL_ID, max_length: int = config.SENTIMENT_MAX_TOKENS):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise SentimentError("Sentiment dependencies are not installed (see requirements-ml.txt).") from exc
        self._torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        except Exception as exc:  # noqa: BLE001 - network/cache/format errors from the hub
            logger.warning("Sentiment model could not be loaded (%s)", type(exc).__name__)
            raise SentimentError(
                "The sentiment model could not be downloaded or loaded. Check network access to "
                "huggingface.co or pre-download the model (see README)."
            ) from exc
        self.model.eval()
        self.max_length = max_length
        self.label_map = normalize_label_map(self.model.config.id2label)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - needs real model
        enc = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        with self._torch.inference_mode():
            logits = self.model(**enc).logits
        return self._torch.softmax(logits, dim=-1).cpu().numpy()


@dataclass
class SentimentRun:
    results: pd.DataFrame  # indexed like the input rows, RESULT_COLUMNS
    analysed: int
    failed: int
    skipped: int
    not_analysed_limit: int
    model_id: str


def analyse(
    df: pd.DataFrame,
    model: SentimentModel,
    *,
    batch_size: int = config.SENTIMENT_BATCH_SIZE,
    max_messages: int = config.SENTIMENT_MAX_MESSAGES,
    model_id: str = MODEL_ID,
    progress: Callable[[float], None] | None = None,
) -> SentimentRun:
    """Classify text messages in ``df``; result rows align with ``df.index``."""
    results = pd.DataFrame(index=df.index, columns=RESULT_COLUMNS, dtype=object)
    results["status"] = "skipped"
    prepared = df["text"].where(df["message_type"] == "text", "").map(preprocess)
    eligible_idx = prepared.index[prepared.str.strip().str.len() > 0]
    limited = 0
    if len(eligible_idx) > max_messages:
        limited = len(eligible_idx) - max_messages
        results.loc[eligible_idx[:limited], "status"] = "not analysed (limit)"
        eligible_idx = eligible_idx[limited:]  # keep the most recent messages (df is chronological)

    label_map = model.label_map
    order = [k for k in sorted(label_map)]
    col_for = {label_map[k]: k for k in order}

    def store(idx: Sequence[object], probs: np.ndarray) -> None:
        probs = np.asarray(probs, dtype=float)
        if probs.shape != (len(idx), len(order)):
            raise ValueError("unexpected output shape")
        neg, neu, pos = (probs[:, col_for[c]] for c in CLASSES)
        best = probs.argmax(axis=1)
        results.loc[idx, "status"] = "ok"
        results.loc[idx, "label"] = [label_map[order[b]] for b in best]
        results.loc[idx, "confidence"] = probs.max(axis=1)
        results.loc[idx, "p_negative"] = neg
        results.loc[idx, "p_neutral"] = neu
        results.loc[idx, "p_positive"] = pos
        results.loc[idx, "polarity"] = pos - neg

    total = len(eligible_idx)
    failures = 0
    for start in range(0, total, batch_size):
        batch_idx = list(eligible_idx[start : start + batch_size])
        texts = prepared.loc[batch_idx].tolist()
        try:
            store(batch_idx, model.predict_proba(texts))
        except Exception as exc:  # noqa: BLE001 - isolate model failures per item
            logger.warning("Sentiment batch failed (%s); retrying items individually", type(exc).__name__)
            for one_idx, one_text in zip(batch_idx, texts, strict=True):
                try:
                    store([one_idx], model.predict_proba([one_text]))
                except Exception as item_exc:  # noqa: BLE001
                    failures += 1
                    results.loc[one_idx, "status"] = "error"
                    logger.warning("Sentiment item failed (%s)", type(item_exc).__name__)
        if progress is not None:
            progress(min(1.0, (start + len(batch_idx)) / max(total, 1)))

    for col in RESULT_COLUMNS[2:]:
        results[col] = pd.to_numeric(results[col], errors="coerce")
    return SentimentRun(
        results=results,
        analysed=int((results["status"] == "ok").sum()),
        failed=failures,
        skipped=int((results["status"] == "skipped").sum()),
        not_analysed_limit=limited,
        model_id=model_id,
    )


def label_distribution(df: pd.DataFrame, run: SentimentRun) -> pd.DataFrame:
    """Per-participant counts and shares of predicted labels (ok rows only)."""
    ok = run.results[run.results["status"] == "ok"]
    if ok.empty:
        return pd.DataFrame(columns=["participant", "label", "messages", "share_pct"])
    joined = pd.DataFrame({"participant": df.loc[ok.index, "sender"].astype(str), "label": ok["label"]})
    counts = joined.groupby(["participant", "label"]).size().rename("messages").reset_index()
    totals = counts.groupby("participant")["messages"].transform("sum")
    counts["share_pct"] = (100 * counts["messages"] / totals).round(1)
    counts["label"] = pd.Categorical(counts["label"], categories=list(CLASSES), ordered=True)
    return counts.sort_values(["participant", "label"]).reset_index(drop=True)


def monthly_summary(df: pd.DataFrame, run: SentimentRun) -> pd.DataFrame:
    """Per-month mean class probabilities, mean polarity index and n."""
    ok = run.results[run.results["status"] == "ok"]
    cols = ["month", "n", "p_negative", "p_neutral", "p_positive", "polarity"]
    if ok.empty:
        return pd.DataFrame(columns=cols)
    month = df.loc[ok.index, "timestamp"].dt.to_period("M").dt.to_timestamp()
    frame = ok[["p_negative", "p_neutral", "p_positive", "polarity"]].assign(month=month.values)
    out = frame.groupby("month").agg(
        n=("polarity", "size"),
        p_negative=("p_negative", "mean"),
        p_neutral=("p_neutral", "mean"),
        p_positive=("p_positive", "mean"),
        polarity=("polarity", "mean"),
    )
    return out.reset_index()[cols]
