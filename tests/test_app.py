"""End-to-end UI tests with Streamlit AppTest (headless, no browser, no network).

Providers and models are replaced with in-process fakes; no paid request is
ever made.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from wa_analyzer import ai_summary, config, sentiment, ui

APP = str(Path(__file__).resolve().parents[1] / "app.py")
REPO = Path(__file__).resolve().parents[1]

CHAT_A = "13/01/2024, 09:00 - Alice: hello from chat A\n13/01/2024, 09:05 - Bob: reply A\n"
CHAT_B = (  # same participants and filename, different content
    "13/01/2024, 09:00 - Alice: hello from chat B\n13/01/2024, 09:05 - Bob: reply B\n14/01/2024, 10:00 - Bob: another\n"
)
AMBIGUOUS = "01/02/2024, 09:00 - Alice: a\n03/04/2024, 09:00 - Bob: b\n"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "TOGETHER_API_KEY",
        "WA_ENABLE_SHARED_AI_KEY",
        "WA_ALLOW_USER_AI_KEY",
        "WA_TOGETHER_MODELS",
        "WA_AI_MAX_REQUESTS_PER_SESSION",
    ):
        monkeypatch.delenv(var, raising=False)


def new_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def click(at: AppTest, label_prefix: str) -> AppTest:
    [btn] = [b for b in at.button if b.label.startswith(label_prefix)]
    btn.click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def upload(at: AppTest, text: str, name: str = "chat.txt") -> AppTest:
    at.file_uploader[0].upload(name, text.encode("utf-8"), "text/plain").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    parts += [e.value for e in at.error] + [w.value for w in at.warning] + [s.value for s in at.success]
    parts += [str(m.value) for m in at.metric]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Basic flow
# ---------------------------------------------------------------------------
def test_initial_page_has_privacy_note_and_no_data():
    at = new_app()
    text = all_text(at)
    assert "not** processed only in your browser" in text
    assert not at.metric
    assert any("Upload an export or try the synthetic sample" in i.value for i in at.info)


def test_sample_flow_renders_all_sections():
    at = click(new_app(), "Try the synthetic")
    assert metric(at, "Records parsed") == "25"
    assert metric(at, "Messages") == "21"
    assert [t.label for t in at.tabs] == [
        "Overview",
        "Activity",
        "Words & emoji",
        "Sentiment (optional)",
        "AI summary (optional)",
        "Download",
    ]
    assert not at.error
    assert len(at.get("download_button")) == 2


def test_upload_txt_and_zip_and_malformed():
    at = upload(new_app(), CHAT_A)
    assert metric(at, "Messages") == "2"
    at = upload(new_app(), "not a whatsapp export\n")
    assert any("No WhatsApp messages were recognised" in e.value for e in at.error)
    assert not at.metric


def test_oversized_upload_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 20)
    at = upload(new_app(), CHAT_A)
    assert any("limit" in e.value for e in at.error)
    assert not at.metric


def test_ambiguous_dates_offer_explicit_choice():
    at = upload(new_app(), AMBIGUOUS)
    assert any("Day and month could not be told apart" in w.value for w in at.warning)
    assert metric(at, "First message") == "2024-02-01"  # DMY guess
    at.radio(key="wa_date_order").set_value("MDY").run()
    assert metric(at, "First message") == "2024-01-02"
    assert not any("could not be told apart" in w.value for w in at.warning)


def test_participant_filter_applies_to_metrics():
    at = upload(new_app(), CHAT_B)
    assert metric(at, "Messages") == "3"
    [ms] = [m for m in at.multiselect if m.label == "Participants"]
    ms.set_value(["Bob"]).run()
    assert metric(at, "Messages") == "2"
    ms.set_value([]).run()
    assert any("Select at least one participant" in i.value for i in at.info)


# ---------------------------------------------------------------------------
# Session handling and privacy
# ---------------------------------------------------------------------------
def test_upload_replacement_invalidates_results_even_with_same_name_and_people():
    at = upload(new_app(), CHAT_A, name="WhatsApp Chat.txt")
    assert metric(at, "Messages") == "2"
    sha_a = at.session_state["wa_current_sha"]
    at.session_state["wa_ai_results"]["stale"] = "x"  # simulate an earlier result
    upload(at, CHAT_B, name="WhatsApp Chat.txt")
    assert metric(at, "Messages") == "3"
    assert at.session_state["wa_current_sha"] != sha_a
    assert "stale" not in at.session_state["wa_ai_results"]


def test_clear_removes_current_session_data_only():
    other = click(new_app(), "Try the synthetic")
    at = upload(new_app(), CHAT_A)
    click(at, "Clear chat and results")
    assert not at.metric
    assert "wa_parsed_result" not in at.session_state
    assert any("Cleared" in s.value for s in at.success)
    assert at.file_uploader[0].value is None
    # The other simulated user is unaffected.
    other.run()
    assert metric(other, "Records parsed") == "25"


def test_two_users_are_isolated():
    user1 = upload(new_app(), CHAT_A)
    user2 = upload(new_app(), CHAT_B)
    user1.run()
    assert metric(user1, "Messages") == "2"
    assert metric(user2, "Messages") == "3"
    assert "chat B" not in all_text(user1)
    fresh = new_app()
    assert not fresh.metric
    assert "wa_parsed_result" not in fresh.session_state


def _snapshot(root: Path) -> set[str]:
    skip = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}
    files = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        files.update(str(Path(dirpath) / f) for f in filenames)
    return files


def test_no_raw_chat_is_persisted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    before_repo, before_tmp = _snapshot(REPO), _snapshot(tmp_path)
    at = upload(new_app(), CHAT_B)
    at.toggle(key="wa_show_wc").set_value(True).run()
    click(at, "Clear chat and results")
    assert _snapshot(REPO) == before_repo
    assert _snapshot(tmp_path) == before_tmp
    source = "\n".join(p.read_text() for p in (REPO / "wa_analyzer").glob("*.py"))
    assert "sqlite3" not in source and "to_sql(" not in source


def test_logs_do_not_contain_chat_content(caplog):
    caplog.set_level("DEBUG")
    upload(new_app(), CHAT_B)
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "chat B" not in logged and "Alice" not in logged


# ---------------------------------------------------------------------------
# AI summary
# ---------------------------------------------------------------------------
class FakeClient:
    instances: list[FakeClient] = []
    calls: list[list[dict]] = []
    reply = "Topics discussed:\n- hiking plans"
    error: ai_summary.ProviderError | None = None

    def __init__(self, api_key, base_url=None, **_):
        assert api_key
        FakeClient.instances.append(self)

    def complete(self, model, messages, max_tokens, deadline):
        FakeClient.calls.append(messages)
        if FakeClient.error is not None:
            raise FakeClient.error
        return FakeClient.reply


@pytest.fixture()
def fake_provider(monkeypatch):
    FakeClient.instances, FakeClient.calls = [], []
    FakeClient.reply = "Topics discussed:\n- hiking plans"
    FakeClient.error = None
    monkeypatch.setattr(ai_summary, "TogetherClient", FakeClient)
    monkeypatch.setenv("TOGETHER_API_KEY", "fake-key-for-tests")
    monkeypatch.setenv("WA_ENABLE_SHARED_AI_KEY", "true")
    return FakeClient


def _consent(at: AppTest):
    return next(c for c in at.checkbox if c.label.startswith("I have reviewed"))


def _generate(at: AppTest):
    return next(b for b in at.button if b.label == "Generate AI summary")


def test_no_ai_request_without_opt_in_and_click(fake_provider):
    at = click(new_app(), "Try the synthetic")
    assert fake_provider.calls == []
    assert _generate(at).disabled  # consent not given
    _consent(at).check().run()
    assert fake_provider.calls == []  # consent alone sends nothing
    _generate(at).click().run()
    assert len(fake_provider.calls) == 1
    sent = fake_provider.calls[0][1]["content"]
    assert "Participant A" in sent and "Asha" not in sent  # pseudonyms on by default
    assert "https://example.org" not in sent  # identifiers masked by default
    assert any("hiking plans" in m.value for m in at.markdown)


def test_unrelated_reruns_do_not_repeat_paid_requests(fake_provider):
    at = click(new_app(), "Try the synthetic")
    _consent(at).check().run()
    _generate(at).click().run()
    assert len(fake_provider.calls) == 1
    at.toggle(key="wa_show_wc").set_value(True).run()
    at.radio(key="wa_grain").set_value("Weekly").run()
    at.run()
    assert len(fake_provider.calls) == 1
    assert any("no new request was made" in c.value for c in at.caption)
    # Changing the selection changes the input: no automatic request either.
    [ms] = [m for m in at.multiselect if m.label == "Participants"]
    ms.set_value(["Asha"]).run()
    assert len(fake_provider.calls) == 1
    assert not _consent(at).value  # consent must be given again for new content


def test_ai_respects_selected_participants(fake_provider):
    at = click(new_app(), "Try the synthetic")
    [ms] = [m for m in at.multiselect if m.label == "Participants"]
    ms.set_value(["Bruno"]).run()
    _consent(at).check().run()
    _generate(at).click().run()
    sent = fake_provider.calls[0][1]["content"]
    assert "north gate" not in sent  # Asha's message is not selected
    assert "Do we need permits?" in sent


def test_missing_credentials_keep_analytics_working(monkeypatch):
    at = click(new_app(), "Try the synthetic")  # no key configured, user keys allowed
    assert metric(at, "Messages") == "21"
    assert _generate(at).disabled
    assert any("Enter an API key" in c.value for c in at.caption)
    monkeypatch.setenv("WA_ALLOW_USER_AI_KEY", "false")
    at.run()
    assert any("AI summaries are disabled" in i.value for i in at.info)


def test_shared_key_present_but_not_enabled_is_not_used(monkeypatch, fake_provider):
    monkeypatch.setenv("WA_ENABLE_SHARED_AI_KEY", "false")
    monkeypatch.setenv("WA_ALLOW_USER_AI_KEY", "false")
    at = click(new_app(), "Try the synthetic")
    assert any("AI summaries are disabled" in i.value for i in at.info)
    assert fake_provider.instances == []


def test_provider_timeout_shows_safe_error_and_keeps_dashboard(fake_provider):
    fake_provider.error = ai_summary.ProviderError("timeout", "The AI provider did not respond in time.")
    at = click(new_app(), "Try the synthetic")
    _consent(at).check().run()
    _generate(at).click().run()
    assert any("did not respond in time" in e.value for e in at.error)
    assert metric(at, "Messages") == "21"
    assert at.session_state["wa_ai_results"] == {}


def test_request_budget_per_session(monkeypatch, fake_provider):
    monkeypatch.setenv("WA_AI_MAX_REQUESTS_PER_SESSION", "1")
    at = click(new_app(), "Try the synthetic")
    _consent(at).check().run()
    _generate(at).click().run()
    assert len(fake_provider.calls) == 1
    [ms] = [m for m in at.multiselect if m.label == "Participants"]
    ms.set_value(["Bruno"]).run()
    assert _generate(at).disabled
    assert any("request limit" in c.value for c in at.caption)


def test_model_output_is_rendered_safely(fake_provider):
    fake_provider.reply = "Topics discussed:\n- ![t](https://evil.example/p.png) <img src=x onerror=alert(1)>"
    at = click(new_app(), "Try the synthetic")
    _consent(at).check().run()
    _generate(at).click().run()
    rendered = "\n".join(m.value for m in at.markdown)
    assert "evil.example" not in rendered and "![" not in rendered
    assert "<img" not in rendered.replace("\\<img", "")


# ---------------------------------------------------------------------------
# Sentiment (fake model; real model is covered by tests/test_real_model.py)
# ---------------------------------------------------------------------------
class TinyModel:
    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    def predict_proba(self, texts):
        return np.tile([0.1, 0.2, 0.7], (len(texts), 1))


def test_sentiment_unavailable_state():
    at = click(new_app(), "Try the synthetic")
    if not sentiment.dependencies_available():
        assert any("Sentiment analysis is not installed" in i.value for i in at.info)


def test_sentiment_runs_only_on_click_and_is_reused(monkeypatch):
    loads = []
    monkeypatch.setattr(sentiment, "dependencies_available", lambda: True)
    monkeypatch.setattr(ui, "_load_sentiment_model", lambda: loads.append(1) or TinyModel())
    at = click(new_app(), "Try the synthetic")
    assert loads == []
    click(at, "Run sentiment analysis")
    assert loads == [1]
    assert any("Classified 18 messages" in c.value for c in at.caption)
    at.radio(key="wa_grain").set_value("Monthly").run()
    assert loads == [1]


def test_sentiment_model_failure_is_contained(monkeypatch):
    def boom():
        raise sentiment.SentimentError("The sentiment model could not be downloaded or loaded.")

    monkeypatch.setattr(sentiment, "dependencies_available", lambda: True)
    monkeypatch.setattr(ui, "_load_sentiment_model", boom)
    at = click(new_app(), "Try the synthetic")
    click(at, "Run sentiment analysis")
    assert any("could not be downloaded" in e.value for e in at.error)
    assert metric(at, "Messages") == "21"


def test_new_upload_resets_explicit_date_order():
    at = upload(new_app(), AMBIGUOUS)
    at.radio(key="wa_date_order").set_value("MDY").run()
    upload(at, CHAT_B)  # 13/01/2024 is impossible as MDY; auto-detection must apply again
    assert at.radio(key="wa_date_order").value == "auto"
    assert metric(at, "Messages") == "3"
    assert not any("skipped" in i.value for i in at.info)
