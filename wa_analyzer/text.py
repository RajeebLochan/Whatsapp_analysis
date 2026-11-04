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
import string
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


def count_links(text: str) -> int:
    return len(URL_RE.findall(text or ""))


_WORD_TOKEN_RE = re.compile(r"\S*?[^\W_]\S*")  # a whitespace token containing a letter or digit
_ASCII_PUNCT = string.punctuation


def _drop_links(text: str) -> str:
    lowered = text.lower()
    if "http" in lowered or "www." in lowered:
        return URL_RE.sub(" ", text)
    return text


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(_WORD_TOKEN_RE.findall(_drop_links(text)))


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens with surrounding punctuation/emoji removed.

    Handles combining marks (e.g. Devanagari/Gujarati vowel signs) correctly
    because it never splits inside a whitespace-separated token.
    """
    if not text:
        return []
    out: list[str] = []
    for raw in _drop_links(text).split():
        if raw.isascii():  # fast path for the common case
            tok = raw.strip(_ASCII_PUNCT).lower()
            if tok and any(ch.isalpha() for ch in tok):
                out.append(tok)
            continue
        tok = _strip_edges(raw).casefold()
        if tok and any(unicodedata.category(ch)[0] == "L" for ch in tok):
            out.append(tok)
    return out


# Code points that can start or form an emoji sequence (superset; exact matching is
# done by the ``emoji`` package). Lets text without emoji, e.g. Hindi or Gujarati
# script, skip the slower full scan.
_EMOJI_CANDIDATE_RE = re.compile("[\u00a9\u00ae\u203c-\u3299\U0001f000-\U0001faff\U000e0020-\U000e007f\ufe0f]")


def extract_emoji(text: str) -> list[str]:
    if not text or text.isascii() or not _EMOJI_CANDIDATE_RE.search(text):
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
