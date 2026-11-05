# Audit report — WhatsApp Chat Analyzer

- Repository: `RajeebLochan/Whatsapp_analysis` (public; created 2025-07-15; 0 forks at audit time)
- Audited commit: `459cf1ac7a7b72a4d541b96127c9e8cdcc374395` (HEAD of `main` when the work started; verified)
- Audit date: 2026-11-04
- Scope: `app.py`, `helper.py`, `preprcosseing.py`, `requirements.txt`, `style.css`,
  `.github/workflows/main_whatsappanalyze.yml`, `.devcontainer/devcontainer.json`,
  tracked `.env` and `uploaded_files.db`, and git history.
- No repository instructions (`CLAUDE.md`, `AGENTS.md`, contributing guide) existed.

## Method and evidence classes

| Class | Meaning |
|---|---|
| **Reproduced** | Demonstrated by running the original code on synthetic input (commands in `VALIDATION_REPORT.md`). |
| **Confirmed (data)** | Verified by inspecting repository files/history with scripts that printed only shapes, counts and lengths — never secret values or chat text. |
| **Source-confirmed** | Evident from reading the code; not executed. |
| **Unverified** | Plausible risk that could not be verified from this environment. |

Secrets were never printed, used or tested. Chat content in the database was never
read or displayed; only row counts, free-page counts and the number of
timestamp-shaped byte patterns were computed.

Research constraints: this environment's network policy blocked `docs.together.ai`,
`huggingface.co`, `docs.streamlit.io` and `owasp.org`. Streamlit behaviour was checked
against the installed 1.66.1 source; Together, Hugging Face and OWASP facts come from
web search results (including official-domain snippets) and are flagged where relevant.

## Findings

Severity: Critical / High / Medium / Low. Line numbers refer to the audited commit.

### Privacy, secrets and data retention

| ID | Sev. | Finding | Evidence | Files | Class |
|---|---|---|---|---|---|
| F-01 | Critical | An API key (64 hex characters, the format of a `TOGETHER_API_KEY`) was committed in `.env` in commit `88e21e7` (2025-07-15) and remained in 5 commits until `3fa0a62` (2025-07-18) replaced it with a 12-character placeholder. The repository is public, so the key must be treated as compromised. The current tree still tracked `.env`. | History scan: `.env` value length/charset per commit; only `88e21e7` adds a 64-hex string. | `.env` | Confirmed (data) |
| F-02 | Critical | Every upload's filename and full raw chat text was saved automatically to SQLite (`save_to_db`, `app.py:34-42`, called at `app.py:200`) and `uploaded_files.db` was committed. Historical versions held 3–9 live rows of WhatsApp-export-shaped text. At HEAD the table had 0 rows, but 288 free pages still contained ~15,600 timestamp-shaped patterns: deleted rows remain recoverable from the file and from every historical copy. Deployed instances (Streamlit Cloud/Azure) would also have written this file on their servers. | `sqlite3` read-only: row counts, `PRAGMA freelist_count`, regex hit counts on raw bytes. | `app.py`, `uploaded_files.db` | Confirmed (data) |
| F-03 | High | README claimed "All data processing happens locally on your machine", "Chat data is not stored or transmitted to external servers". All three were false (server-side processing, SQLite persistence, automatic Together AI transmission). | README "Privacy and Security". | `README.md` | Source-confirmed |
| F-04 | High | On every analysis run the **entire raw upload** (ignoring the user filter) was sent to Together AI without notice or consent, with a prompt asking the model to profile the person's personality, values and identity and to "guess who is the person". | `app.py:1242-1276` | `app.py` | Source-confirmed |
| F-05 | High | The AI call ran on every rerun (any widget change re-executed the script), repeating paid requests with no caching, timeouts, retries or limits. | `app.py:1261` executes unconditionally inside the analysis block | `app.py` | Source-confirmed |
| F-06 | High | A missing `TOGETHER_API_KEY` raised `ValueError`, crashing the page after all other analytics had rendered. Raw exception text was shown to users elsewhere (`st.error(f"... {e}")`). | `app.py:1245-1246`, `app.py:489` | `app.py` | Source-confirmed |
| F-07 | Medium | No upload size or processing limits; non-UTF-8 files crashed with `UnicodeDecodeError`. | `app.py:191`; reproduced decode error | `app.py` | Reproduced |
| F-08 | Medium | Devcontainer started Streamlit with `--server.enableCORS false --server.enableXsrfProtection false`. | `devcontainer.json` | `.devcontainer/` | Source-confirmed |
| F-09 | Low | Compiled `__pycache__/*.pyc` files were committed in history (no secrets found in them by the key-pattern scan). | `git log --diff-filter=A` | history | Confirmed (data) |

### Parsing

| ID | Sev. | Finding | Evidence | Files | Class |
|---|---|---|---|---|---|
| F-10 | High | The first message was discarded (`re.split(...)[2:]` and `findall(...)[1:]`): a 3-message chat yielded 2 rows; a single-message chat yielded 0. It only "worked" when the first line was the encryption notice. | Synthetic run | `preprcosseing.py:10,32` | Reproduced |
| F-11 | High | Only US `M/D/YY, h:mm AM` was parsed (`format='%m/%d/%y, %I:%M %p'`): DD/MM/YYYY 24-hour and iOS bracketed exports produced **0 rows** without an error. | Synthetic runs | `preprcosseing.py:7,36` | Reproduced |
| F-12 | High | Messages and timestamps were extracted by two different regexes and zipped by position. A timestamp-like string inside a message body crashed parsing (`ValueError: All arrays must be of the same length`); other variants can silently misalign. | Synthetic run | `preprcosseing.py` | Reproduced |
| F-13 | Medium | Splitting on the first `": "` turned system lines containing colons (e.g. subject changes) into fake senders; media/deleted markers were not classified (only `<Media omitted>` was recognised). | Code reading | `preprcosseing.py:21-28`, `helper.py` | Source-confirmed |
| F-14 | Low | Column `day_name` contained the day of the month (1–31). | Synthetic run | `preprcosseing.py:40` | Reproduced |

### Statistics and interpretation

| ID | Sev. | Finding | Evidence | Files | Class |
|---|---|---|---|---|---|
| F-15 | High | "Avg Messages/Day" divided by the number of unique *timestamps* (to the minute) and ignored the selected user: ten messages on one day showed 1.0. | Synthetic run | `app.py:270` | Reproduced |
| F-16 | Medium | Filters were inconsistent: several sections used the unfiltered `df` (`app.py:341,390,422,436,450`); the whole "Text Level Analysis" ignored the selected user. | Code reading | `app.py` | Source-confirmed |
| F-17 | Medium | "Average response time" was the mean gap between consecutive messages *of the filtered user*, i.e. not a response time; with "All Users" it mixed all gaps including multi-day silences. | `helper.py:average_response_time` | `helper.py` | Source-confirmed |
| F-18 | Medium | Silence gaps were computed after dropping system rows, rows with no text and rows where sentiment failed, so results depended on model success. They were labelled "Who Was Silent" / "who broke the silence", implying responsibility. | `app.py:471,523,541,1171-1227` | `app.py` | Source-confirmed |
| F-19 | Medium | Unsupported interpretations: message counts labelled "Conversation Initiation … who puts more effort into continuing the relationship"; "Emotional Imbalance Meter"; keyword "Topic Segmentation" (study/love/anger/confusion) presented as topic modelling; "relationship dynamics" claims. | `app.py:698-1141, 1224-1227` | `app.py`, `README.md` | Source-confirmed |
| F-20 | Medium | Time grouping errors: month names grouped across years and sorted alphabetically; weekdays sorted alphabetically; ISO week numbers without year; seasons by northern-hemisphere months. | `app.py:797-1022` | `app.py` | Source-confirmed |
| F-21 | Medium | Emoji were counted per code point: 👍🏽 became 👍 + 🏽, 👨‍👩‍👧 became three emoji, and the flag 🇮🇳 was dropped entirely. | Synthetic run | `helper.py:common_emojis` | Reproduced |
| F-22 | Medium | Helpers mutated the caller's DataFrame (`df['user_message'] = …` in `word_cloud`, `common_words`, `common_emojis`; `df['date'] = …` elsewhere) and returned inconsistent types (number vs. DataFrame vs. string "N days"). | Code reading | `helper.py` | Source-confirmed |
| F-23 | Medium | An empty selection crashed the word cloud (`ValueError: We need at least 1 word…`). | Synthetic run | `helper.py:word_cloud` | Reproduced |
| F-24 | Low | Sections labelled "Heatmap" rendered line charts; ~50 repetitive table+line+bar combinations made the page long and hard to read. | `app.py:744-758` and onward | `app.py` | Source-confirmed |
| F-25 | Medium | Stale results: the analysis DataFrame was stored on button click and reused while the selected user stayed the same, so uploading a different file kept showing the previous file's analysis. | `app.py:228-253` | `app.py` | Source-confirmed |

### Sentiment

| ID | Sev. | Finding | Evidence | Files | Class |
|---|---|---|---|---|---|
| F-26 | High | The winning label's *confidence* was plotted as a 0 (negative) – 1 (positive) "sentiment score" with a neutral line at 0.5 and averaged per user/date/topic, so a confident negative (score 0.95) counted as strongly positive. | `app.py:519-636, 738-1128` | `app.py` | Source-confirmed |
| F-27 | Medium | Preprocessing removed emoji and other symbols (`[^\w\s.?!]`), unlike the model card's procedure (mask `@user` and links, keep emoji). | `helper.py:preprocess_text` | `helper.py` | Source-confirmed |
| F-28 | Medium | Inference ran automatically, one message at a time with no batching, length bound or cap; a model-load failure called `st.stop()` and ended the dashboard; a bare `except:` hid errors. | `app.py:479-513` | `app.py` | Source-confirmed |
| F-29 | Medium | The model is not evaluated for WhatsApp/Hinglish/Gujarati, yet the README claimed multi-language support and accurate mood analysis. The model card's fine-tuning languages are Ar, En, Fr, De, Hi, It, Es, Pt. | README; model-card search snippet | `README.md` | Source-confirmed / search |

### External AI

| ID | Sev. | Finding | Evidence | Files | Class |
|---|---|---|---|---|---|
| F-30 | High | Hard-coded model `meta-llama/Llama-4-Scout-17B-16E-Instruct`. Search results from Together's changelog list it as deprecated / no longer available. | Web search (official page not reachable here) | `app.py:1262` | Unverified (secondary evidence) |
| F-31 | High | No separation between instructions and chat data, chat text and model output rendered as Markdown (remote images in output could load in the viewer's browser), no output handling. | `app.py:1251-1275` | `app.py` | Source-confirmed |

### Engineering, dependencies and delivery

| ID | Sev. | Finding | Evidence | Files | Class |
|---|---|---|---|---|---|
| F-32 | Medium | `nltk.download('stopwords')` ran at import time (network access on every cold start). | `helper.py:56` | `helper.py` | Source-confirmed |
| F-33 | Medium | Default requirements pinned `torch` plus 15 `nvidia-*` CUDA wheels and unused packages (`google-generativeai`, `textblob`, `GitPython`, …). `pip-audit` on the pinned set: 23 vulnerable packages, 374 advisory entries (aliases included), e.g. `transformers 4.53.2`, `torch 2.7.1`, `streamlit 1.46.1`, `pillow`, `aiohttp`, `urllib3`, `nltk`. | `pip-audit -r` (no install) | `requirements.txt` | Reproduced |
| F-34 | Low | 1,300-line `app.py` with UI, parsing, statistics, ML and API code mixed; HTML strings with `unsafe_allow_html`; CSS hid Streamlit's header (and with it the sidebar toggle on small screens). | Code reading | `app.py`, `style.css` | Source-confirmed |
| F-35 | Medium | No tests or CI. The Azure workflow deployed on every push to `main` without tests, and `zip release.zip ./* -r` silently omitted dot-directories such as `.streamlit/`. | Workflow | `.github/workflows/` | Source-confirmed |
| F-36 | Low | README described files and features that did not exist (`preprocessor.py`, LDA topic modelling, language detection, predictive analytics, parallel processing, MIT `LICENSE` file). | README vs tree | `README.md` | Source-confirmed |

### Residual / unverified risks

| ID | Risk | Notes |
|---|---|---|
| U-01 | Copies of uploaded chats may exist on deployed servers (Streamlit Community Cloud, Azure App Service) in `uploaded_files.db`, and in any clones of this public repository. | Not accessible from this environment. Owner action in the runbook. |
| U-02 | Whether the exposed key was used by third parties. | Only the owner can check the provider's usage logs. |
| U-03 | Current Together model IDs and data-retention terms. | Official docs unreachable here; defaults are configurable and documented as needing verification. |
| U-04 | Real sentiment-model behaviour with the new code. | `huggingface.co` and the PyTorch CPU index were unreachable; real-model test is provided but could not be run here. |
