# Validation report

Date: 2026-11-04 · Branch: `claude/sleepy-ramanujan-mhm11b` · Base: `459cf1a`
Environment: Linux x86_64, 4 vCPU, CPython 3.11.18 and 3.13.18, Chromium (Playwright build 1243).
All data used in validation is synthetic (`wa_analyzer/sample.py` and inline test chats).

## 1. Summary

| Check | Command | Result |
|---|---|---|
| Lint | `ruff check .` | All checks passed (3.11, 3.13) |
| Format | `ruff format --check .` | 24 files already formatted |
| Unit + AppTest suite | `python -m pytest -q --ignore=tests/browser` | **111 passed, 1 skipped** on 3.11 and on 3.13 (skip = opt-in real-model test) |
| Browser smoke test | `python -m pytest -q tests/browser` | **2 passed** (desktop flow + mobile layout) |
| Dependency consistency | `pip check` / `uv pip check` in clean venvs | All installed packages are compatible (3.11, 3.13) |
| Dependency resolution | `uv pip compile` of `requirements-dev.txt` and `requirements.txt + requirements-ml.txt` for 3.11/3.13 (manylinux x86_64) | Resolves; ML graph → torch 2.15.0, transformers 5.22.1, tokenizers 0.23.3, sentencepiece 0.2.2, numpy 2.4.6 (3.11) / 2.5.4 (3.13) |
| Vulnerability audit | `pip-audit` on installed core+dev env; `pip-audit --no-deps -r` on resolved ML graphs (3.11, 3.13) | No known vulnerabilities found |
| Audit of the original pins | `pip-audit --no-deps -r <requirements.txt at 459cf1a>` | 23 vulnerable packages, 374 advisory entries (aliases included) |
| Secret scan (current tree) | `git grep -nIE "[0-9a-f]{64}\|tgp_v1_\|sk-…"` | No matches |
| Static privacy checks | grep for file writes / SQLite / `unsafe_allow_html` / caches in `wa_analyzer` | No file writes or SQLite; one `unsafe_allow_html` (static CSS); one `st.cache_resource` (model weights only) |

## 2. Test inventory (what each layer validates)

| File | Tests | Validates | Network/model |
|---|---:|---|---|
| `tests/test_parsing.py` | 31 | Android/iOS formats, 12/24 h, seconds, years, separators, NBSP/BOM/bidi marks, multiline, colons, Unicode/phone senders, U+2028, system/media/deleted/edited, ambiguous dates & explicit order, invalid timestamps, stable sorting, limits, encodings, ZIP guards, first-message preservation and alignment | none |
| `tests/test_analytics.py` | 12 | Hand-calculated overview, per-participant, daily/hourly/weekday/heatmap, year-aware monthly/weekly, response gaps before filtering, idle-gap sessions, silence gaps, filter consistency, compound emoji, non-mutation, empty inputs | none |
| `tests/test_export_text.py` | 15 | CSV formula-injection protection, ZIP/report without message text, pseudonyms, word/link/emoji/tokeniser definitions (incl. Devanagari/Gujarati) | none |
| `tests/test_sentiment.py` | 9 | Label-map normalisation, model-card preprocessing, confident negative stays negative, confidence ≠ polarity, row alignment with skips/failures, per-item retry, message cap, aggregations | fake model |
| `tests/test_ai_summary.py` | 22 | Pseudonyms/redaction, delimiter neutralisation, bounded chunking, cache keys, request shape (no tools, timeouts), auth/model/400 errors, 429/5xx retries with Retry-After, timeouts, connection errors, interrupted/incomplete streams, empty output, deadline, map-reduce request counting, budget, safe rendering, shared-key default off | fake HTTP session |
| `tests/test_app.py` | 22 | Streamlit AppTest of the real `app.py`: privacy note, sample and upload flows, malformed/oversize uploads, ambiguous-date choice, filters, upload replacement (same name and people), clear, two-user isolation, no files written, logs without chat content, AI opt-in/consent/click, no repeated requests on unrelated reruns, filtered AI input, missing keys, timeouts, request budget, safe output, sentiment unavailable/lazy/failure | fakes only |
| `tests/browser/test_browser.py` | 2 | Real Chromium against `streamlit run app.py`: sample flow, Plotly charts render, a heatmap layer is drawn, AI button disabled without consent, clear works, no JS page errors; 390 px mobile viewport has no horizontal overflow and tabs remain reachable | local server only |
| `tests/test_real_model.py` | 1 | Real model download, label mapping, CPU batch inference, two obvious English sentences on the expected side. **Not an accuracy evaluation.** | needs huggingface.co (skipped here) |

## 3. Commands used

```bash
uv venv -p 3.11 /tmp/clean3.11 && uv pip install -p /tmp/clean3.11/bin/python -r requirements-dev.txt
uv venv -p 3.13 /tmp/clean3.13 && uv pip install -p /tmp/clean3.13/bin/python -r requirements-dev.txt
uv pip check -p <venv>/bin/python
ruff check . && ruff format --check .
python -m pytest -q --ignore=tests/browser
PLAYWRIGHT_CHROMIUM_EXECUTABLE=/opt/pw-browsers/chromium-1243/chrome-linux64/chrome python -m pytest -q tests/browser
uv pip compile --python-version 3.13 --python-platform x86_64-manylinux_2_28 requirements.txt requirements-ml.txt -o ml.lock
pip-audit                                   # installed core+dev environment
pip-audit --no-deps --disable-pip -r ml.lock
python scripts/benchmark.py
```

Reproduction of original defects (run against `459cf1a`'s `preprcosseing.py` with
synthetic input; outputs abbreviated):

```text
3-message US chat              -> 2 rows (first message lost)
single-message chat            -> 0 rows
DD/MM/YYYY 24-hour chat        -> 0 rows
iOS bracketed chat             -> 0 rows
timestamp text inside a message-> ValueError: All arrays must be of the same length
10 messages on one day         -> "Avg Messages/Day" 1.0
emoji in "👍🏽🇮🇳👨‍👩‍👧"            -> ['👍', '🏽', '👨', '👩', '👧'] (flag dropped)
empty word cloud               -> ValueError: We need at least 1 word to plot a word cloud
non-UTF-8 upload               -> UnicodeDecodeError
day_name column                -> [5, 5, 6] (day of month)
```

## 4. Performance (measured)

`python scripts/benchmark.py` on Python 3.13, synthetic Android-style chats with 6
participants. Time is wall-clock without tracing; heap is the tracemalloc peak of a
second, traced run (a lower bound: some native allocations are not traced).

| messages | export size (MB) | parse time (s) | parse peak heap (MB) | core stats time (s) | stats peak heap (MB) |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.1 | 0.06 | 1 | 0.07 | 0 |
| 10,000 | 0.6 | 0.23 | 12 | 0.20 | 1 |
| 100,000 | 6.3 | 1.85 | 123 | 1.10 | 8 |
| 300,000 | 18.9 | 8.40 | 371 | 3.22 | 27 |

Process peak RSS for the whole run (all sizes, interpreter, pandas): 1,213 MB.
"Core stats" = overview, participant table, daily/monthly series, heatmap matrix,
response gaps, sessions, silence gaps, word frequencies and top emoji.

Before the per-upload feature columns and fast paths, the same core statistics took
5.69 s (10k) and 168 s (300k) under tracemalloc; an untraced profile at 20k messages
showed ~1.4 s, dominated by repeated emoji extraction and tokenisation. In the app,
parsing runs once per upload/date order and statistics are memoised per session
and selection, so ordinary widget changes do not repeat them.

Estimate (not measured end-to-end in a browser): a 20 MB upload needs roughly
0.4–0.5 GB of transient memory per session while parsing; size hosts accordingly.

## 5. Security and privacy checks

- History scan (values never printed): `.env` held a 64-hex value in commits
  `88e21e7`…`d95dff9`; `3fa0a62` replaced it with a 12-character placeholder. All 17
  commits reachable from `459cf1a` contain `.env` and/or `uploaded_files.db`.
- `uploaded_files.db` at `459cf1a`: 0 live rows, 288 free pages, ~15,600
  timestamp-shaped byte patterns (deleted chat content recoverable). Earlier commits:
  3–9 live rows.
- Repository visibility (GitHub search API): public, 0 forks.
- Current tree: `.env` and `uploaded_files.db` untracked and ignored; no key-like
  strings; `.env.example`/`secrets.toml.example` contain placeholders only.
- AppTest verifies that a full session (upload, word cloud, clear) creates no files in
  the repository or the temp directory, and that log records contain no names or text.
- Ruff `S` (flake8-bandit) rules enabled for application code: no findings.

## 6. UI review

- Checked in a real browser (headless Chromium via Playwright) at 1366×900 and
  390×844: screenshots `desktop-overview.png`, `desktop-heatmap.png`, `mobile.png`
  are written to `tests/browser/artifacts/` (git-ignored) on each run and uploaded by CI.
- Verified visually: step headings, privacy note beside the upload, parsing summary,
  filters, tabs, a true weekday × hour heatmap, mobile tab scrolling and the sidebar
  toggle remaining available.
- Accent contrast (WCAG ratio, computed): light-theme primary `#0F7B5C` 5.2:1 on white;
  dark-theme primary `#1A8F68` 4.7:1 on Streamlit's dark background. Sentiment colours
  use a blue/grey/red diverging scheme, checked with a palette validator: worst
  adjacent colour-blind ΔE 8.6 (pass), all three ≥ 3:1 against a light surface; the
  grey midpoint fails the *categorical* chroma floor by design (a neutral midpoint
  is intended to read as grey). Colours are always paired with legends/labels.
- Not done: screen-reader testing, colour-contrast audit of Streamlit's built-in
  components, dark-mode screenshots, cross-browser (Firefox/Safari) runs.

## 7. Limitations

1. **Real sentiment model not executed.** `huggingface.co` and
   `download.pytorch.org` were blocked by this environment's network policy. Sentiment
   logic is validated with fake models; `tests/test_real_model.py` and the manual
   `real-model.yml` workflow are provided but did not run here. No accuracy evaluation
   exists for WhatsApp, Hinglish, Gujarati or other informal text.
2. **Together AI behaviour validated only against mocks.** No real or paid request
   was made, and the exposed key was never used. `docs.together.ai` was unreachable, so
   the default model IDs (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`) and the
   deprecation of Llama-4-Scout come from web-search results, not the official pages.
   Reasoning models may spend output tokens on reasoning; the app reports empty output
   explicitly.
3. **Documentation research** for Streamlit relied on the installed 1.66.1 source
   (APIs such as `width="stretch"`, `max_upload_size`, `on_click="ignore"`,
   `AppTest.file_uploader.upload`), because `docs.streamlit.io` was unreachable.
4. **GitHub Actions workflows were not run on GitHub** from this environment; their
   steps were executed locally where possible (lint, tests, audit, zip contents).
5. **Parser coverage** follows documented/common export shapes; system-event detection
   is English-only, and exports from unusual locales (e.g. non-Latin digits) are not
   supported and will be reported as unrecognised.
6. **Memory**: tracemalloc figures are lower bounds; concurrent large uploads on small
   hosts can exhaust memory.
