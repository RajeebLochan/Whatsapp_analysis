"""Text helpers: links, words, stop words and emoji.

Definitions used throughout the app:

- **Link**: a substring starting with ``http://``, ``https://`` or ``www.``.
- **Word**: a whitespace-separated token (links removed) that contains at least
  one letter or digit. Emoji-only and punctuation-only tokens are not words.
- **Emoji**: a complete emoji sequence as defined by the ``emoji`` package's
  Unicode data, so 👍🏽 (skin tone), 🇮🇳 (flag) and 👨‍👩‍👧 (ZWJ family) each count
  as one. The text-style symbols ©, ® and ™ are excluded.

Stop words are a small built-in English list (no NLTK download is needed).
Other languages, including Hindi, Hinglish and Gujarati, are tokenised
correctly but their function words are not filtered.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

import emoji as emoji_lib

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EXCLUDED_EMOJI = {"©", "®", "™", "©️", "®️", "™️"}

# Compact English stop-word list (based on common NLP lists), lower-case.
STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "can't",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "he'd",
        "he'll",
        "he's",
        "her",
        "here",
        "here's",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "how's",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "that's",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "there's",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "were",
        "weren't",
        "what",
        "what's",
        "when",
        "when's",
        "where",
        "where's",
        "which",
        "while",
        "who",
        "who's",
        "whom",
        "why",
        "why's",
        "will",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "also",
        "just",
        "get",
        "got",
        "im",
        "ive",
        "dont",
        "cant",
        "u",
        "ur",
        "ok",
        "okay",
        "yes",
        "yeah",
    ]
)


def _strip_edges(token: str) -> str:
    start, end = 0, len(token)
    while start < end and unicodedata.category(token[start])[0] in "PSZC":
        start += 1
    while end > start and unicodedata.category(token[end - 1])[0] in "PSZC" and token[end - 1] not in "‍":
        end -= 1
    return token[start:end]


def _has_alnum(token: str) -> bool:
    return any(unicodedata.category(ch)[0] in "LN" for ch in token)


def count_links(text: str) -> int:
    return len(URL_RE.findall(text or ""))


def count_words(text: str) -> int:
    if not text:
        return 0
    return sum(1 for tok in URL_RE.sub(" ", text).split() if _has_alnum(tok))


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens with surrounding punctuation/emoji removed.

    Handles combining marks (e.g. Devanagari/Gujarati vowel signs) correctly
    because it never splits inside a whitespace-separated token.
    """
    if not text:
        return []
    out: list[str] = []
    for raw in URL_RE.sub(" ", text).split():
        tok = _strip_edges(raw).casefold()
        if tok and any(unicodedata.category(ch)[0] == "L" for ch in tok):
            out.append(tok)
    return out


def extract_emoji(text: str) -> list[str]:
    if not text:
        return []
    return [e["emoji"] for e in emoji_lib.emoji_list(text) if e["emoji"] not in _EXCLUDED_EMOJI]


def word_counter(texts: Iterable[str], stopwords: frozenset[str] = STOPWORDS, min_length: int = 2) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(t for t in tokenize(text) if len(t) >= min_length and t not in stopwords)
    return counter


def emoji_counter(texts: Iterable[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(extract_emoji(text))
    return counter
