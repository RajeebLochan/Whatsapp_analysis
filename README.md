# WhatsApp Chat Analyzer

A Streamlit app that turns an exported WhatsApp chat into clear, well-defined
statistics: who wrote how much, when the chat is active, conversation sessions,
response-gap estimates, frequent words and emoji. Optional extras: message-level
sentiment (local model) and a factual AI summary (Together AI, opt-in only).

**Live demo:** [whatsapp-conversation-analyzer.streamlit.app](https://whatsapp-conversation-analyzer.streamlit.app/)
(the hosted demo reflects whatever version its owner has deployed).

## Privacy — read this first

- **Processing happens on the server running the app.** If you use a hosted
  copy, your file is uploaded to that server. It is *not* processed only in your
  browser. Run it locally (below) if you do not want to upload your chat anywhere.
- The app keeps the chat **in memory for your browser session only**. It does
  not write chats, parsed data or summaries to disk or a database, and it does not
  share them between users. **Clear chat and results** removes them from your
  session immediately; they also disappear when the session ends.
- The app's own logs contain only event types and counts, never message text,
  names or prompts. Hosting providers (Streamlit Community Cloud, Azure, …) may keep
  their own request/access logs.
- **Nothing leaves the server unless you ask for an AI summary.** You see exactly
  what will be sent, must tick a consent box for that exact content, and must
  click **Generate AI summary**. The selected messages then go to Together AI
  under its terms. Pseudonyms and masking of e-mails, phone numbers, links and
  @mentions are on by default; they reduce exposure but **do not guarantee
  anonymity**.
- The optional sentiment model is downloaded from Hugging Face on first use; chat
  text is classified locally and is not sent to Hugging Face.
- Downloads contain aggregate statistics only (no message text) and are
  protected against spreadsheet formula injection.

Only analyse chats you are entitled to analyse, and respect the other
participants' privacy.

## Features

| Area | What you get |
|---|---|
| Parsing review | Detected format, date order (with explicit choice when DD/MM vs MM/DD is ambiguous), counts of text/media/deleted/system entries, skipped lines with line numbers, warnings |
| Filters | Participants, date range and session idle gap; applied consistently to every metric, chart, download and AI input |
| Overview | Messages, words, media placeholders, links, active days, calendar span, averages per active day and per calendar day, per-participant table |
| Activity | Daily/weekly/monthly timeline (zero-filled, year-aware), weekday × hour heatmap, sessions and session starters, response-gap estimates, longest quiet periods |
| Words & emoji | Top words (English stop words removed), word cloud, top emoji (compound emoji counted once) |
| Sentiment (optional) | Negative/neutral/positive class shares per participant and month; polarity index explained |
| AI summary (optional) | Topics, explicitly stated decisions, open questions, action items and stated uncertainty |
| Download | ZIP of aggregate CSVs + Markdown report; optional pseudonymised names |

### Metric definitions

- **Message**: a participant entry — text, media placeholder or deleted marker.
  System events (joins, encryption notice, subject changes…) are counted separately.
- **Word**: a whitespace-separated token in a text message that contains a letter or
  digit; links are excluded.
- **Active day**: a date with ≥1 message. **Calendar span**: first to last message
  date, inclusive. Two averages are shown: per active day and per calendar day.
- **Session**: messages with no gap longer than the chosen idle gap (default 1 hour).
  The *session starter* sent the first message after an idle gap — a timing fact,
  not a measure of effort or intent.
- **Response-gap estimate**: time before a message whose sender differs from the
  previous sender, within a session, computed on the full conversation before any
  participant filter. In group chats the previous message is not necessarily the one
  being answered.
- **Quiet period**: a ≥24 h gap in the whole conversation. It says when the chat was
  quiet, not who was responsible.
- Times are shown as exported (the phone's local time). Exports contain no timezone,
  and none is assumed.

## Supported export formats

Export from WhatsApp via *chat → ⋮ / contact name → More → Export chat → Without
media*. Upload the `.txt` file, or the `.zip` that iOS produces (the `_chat.txt`
inside is read; size limits apply to the extracted text too).

| Style | Example (synthetic) |
|---|---|
| Android, US 12-hour | `1/5/24, 9:01 PM - Alice: hi` (also with a narrow no-break space before PM) |
| Android, 24-hour | `05/01/2024, 21:01 - Alice: hi` |
| Dotted / ISO dates | `05.01.24, 21:01 - Alice: hi` · `2024-01-05, 21:01 - Alice: hi` |
| iOS (bracketed, seconds) | `[05/01/2024, 21:01:07] Alice: hi` · `[1/5/24, 9:01:07 PM] Alice: hi` |

Also handled: 2- or 4-digit years; `AM/PM`, `am/pm`, `a.m./p.m.`; optional
seconds; BOMs and directional marks; multi-line messages; colons in messages; phone
numbers and non-Latin names as senders; `<This message was edited>` markers; media
placeholders (`<Media omitted>`, `image omitted`, `<attached: …>`, `… (file attached)`);
deleted-message markers. System-event detection uses English phrases, so system
lines from exports in other languages may be counted as ordinary text from a
“sender”. Limits: 20 MB per file and 300,000 messages (the most recent are kept,
with a warning).

## Quick start

Requires Python 3.11–3.13.

```bash
git clone https://github.com/RajeebLochan/Whatsapp_analysis.git
cd Whatsapp_analysis
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 and click **Try the synthetic sample chat** (invented
data) or upload your own export.

### Optional: sentiment analysis

```bash
pip install "torch>=2.13,<3" --index-url https://download.pytorch.org/whl/cpu   # CPU-only build, avoids CUDA wheels
pip install -r requirements-ml.txt
```

The first run downloads `cardiffnlp/twitter-xlm-roberta-base-sentiment` (~1.1 GB)
into `~/.cache/huggingface`. The model card lists fine-tuning languages Arabic,
English, French, German, Hindi, Italian, Spanish and Portuguese. This project has
**not** evaluated its accuracy on WhatsApp chats; expect errors with informal
language, sarcasm, romanised Hindi/Hinglish, Gujarati and code-mixed text. The app
analyses at most the 3,000 most recent text messages of a selection per run.

### Optional: AI summaries (Together AI)

AI summaries are off until a user opts in. Two ways to provide a key:

1. **Per user (default):** visitors paste their own Together API key; it is kept in
   memory for that session only.
2. **Shared deployment key:** set `TOGETHER_API_KEY` *and*
   `WA_ENABLE_SHARED_AI_KEY=true`. Without the second flag a configured key is
   ignored. Enable a shared key only when the deployment has access control and the
   provider account has spending limits. `WA_AI_MAX_REQUESTS_PER_SESSION` (default 6)
   is a per-session guard, not a spending control.

Settings are read from environment variables first, then Streamlit secrets — see
`.env.example` and `.streamlit/secrets.toml.example` (placeholders only; never commit
real values). The offered models come from `WA_TOGETHER_MODELS` (default
`openai/gpt-oss-120b,openai/gpt-oss-20b`). Together's catalogue changes often (for
example, `meta-llama/Llama-4-Scout-17B-16E-Instruct`, used previously, is listed as
deprecated), so verify IDs against
<https://docs.together.ai/docs/serverless/models> before deploying. Each summary uses
at most 4 chunks of ~6,000 estimated input tokens (most recent messages first) plus
one combining request, 1,200 output tokens per request, 10 s connect / 60 s read
timeouts, a 180 s overall deadline and at most 2 retries for 429/5xx/network errors.

The summary prompt asks only for explicitly stated topics, decisions, questions and
action items, and for stated uncertainty. Chat text is passed as delimited,
untrusted data and the model has no tools. This reduces, but cannot prevent, prompt
injection. Output is shown as escaped plain text with links removed.

## Project structure

```
app.py                     # Streamlit entry point (calls wa_analyzer.ui.main)
wa_analyzer/
  parsing.py               # export parser (replaces the former preprcosseing.py)
  analytics.py             # statistics (pure, non-mutating)
  text.py                  # words, links, emoji, stop words
  sentiment.py             # optional, lazily loaded sentiment model
  ai_summary.py            # opt-in Together AI summaries, redaction, safe rendering
  export.py                # aggregate CSV/report downloads, CSV-injection protection
  config.py                # limits and secret/setting loading
  sample.py                # synthetic sample chat and generator
  ui.py                    # Streamlit interface
tests/                     # pytest + Streamlit AppTest; tests/browser = Playwright
scripts/benchmark.py       # parsing/statistics benchmark on synthetic data
.streamlit/config.toml     # upload limit, XSRF, error-detail and theme settings
```

## Tests and checks

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
python -m pytest -q --ignore=tests/browser        # offline: parser, analytics, AppTest UI, mocked AI/sentiment
python -m playwright install chromium && python -m pytest -q tests/browser   # real-browser smoke test
WA_RUN_REAL_MODEL=1 python -m pytest -q -m real_model                        # needs requirements-ml.txt + network
pip-audit -r requirements.txt
python scripts/benchmark.py
```

All test data is synthetic. The default suite needs no network, no model download
and no API key; the AI provider and sentiment model are replaced by fakes. See
`VALIDATION_REPORT.md` for the latest results.

## Deployment notes

- **Streamlit Community Cloud:** entry point `app.py`; dependencies from
  `requirements.txt` (core only, so sentiment is shown as unavailable unless you add
  the ML extras); secrets via the app's *Secrets* settings.
- **Azure App Service** (`.github/workflows/main_whatsappanalyze.yml`): the
  workflow now runs tests before packaging and includes `.streamlit/config.toml`.
  The App Service startup command must start Streamlit, e.g.
  `python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0`.
- Keep `server.enableXsrfProtection = true`. If uploads fail with HTTP 403 behind a
  proxy or in Codespaces, fix the proxy/origin configuration rather than disabling
  XSRF protection.
- Each session holds its chat in memory; a 20 MB export can peak at roughly 400 MB of
  Python heap while parsing (see `VALIDATION_REPORT.md`). Size the host or lower
  the limits in `wa_analyzer/config.py` and `.streamlit/config.toml` accordingly.

## Security

See `AUDIT_REPORT.md` and `SECURITY_INCIDENT_RUNBOOK.md`. Earlier commits in this
repository contained an API key and a database with uploaded chat content; the
runbook lists the owner-only cleanup steps.

## Author

**Rajeeb Lochan Behera** —
[GitHub](https://github.com/RajeebLochan) ·
[LinkedIn](https://www.linkedin.com/in/rajeeb-lochan/) ·
[X](https://x.com/rajeeb_thedev)

Not affiliated with WhatsApp or Meta. No licence file is included in this
repository; contact the author about reuse.
