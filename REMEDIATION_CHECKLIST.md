# Remediation checklist

Maps each finding in `AUDIT_REPORT.md` to the fix and the tests that guard it.
Status: ✅ fixed and tested · 🟡 fixed, verification limited (reason given) · 👤 owner action required.

Test files: `tests/test_parsing.py` (P), `tests/test_analytics.py` (A),
`tests/test_export_text.py` (E), `tests/test_sentiment.py` (S),
`tests/test_ai_summary.py` (AI), `tests/test_app.py` (APP, Streamlit AppTest),
`tests/browser/test_browser.py` (BR, Playwright), `tests/test_real_model.py` (RM, opt-in).

## Privacy, secrets and data retention

| ID | Fix | Tests / verification | Status |
|---|---|---|---|
| F-01 | `.env` untracked and ignored; `.env.example` and `.streamlit/secrets.toml.example` hold placeholders only; settings read from env vars, then `st.secrets` (`wa_analyzer/config.py`). Key rotation and history purge are owner actions in `SECURITY_INCIDENT_RUNBOOK.md`. | Tree and history scans (VALIDATION_REPORT §5); AI `test_settings_defaults_disable_shared_key` | ✅ tree · 👤 rotation/history |
| F-02 | `save_to_db` and SQLite removed; `uploaded_files.db` untracked; `*.db` ignored; all chat data kept in `st.session_state`; no `st.cache_data` for chat data. Historical copies and deployed DB files need owner action. | APP `test_no_raw_chat_is_persisted`, `test_two_users_are_isolated`, `test_clear_removes_current_session_data_only` | ✅ code · 👤 history/servers |
| F-03 | README and in-app notes rewritten: server-side processing, in-memory session storage, provider logs, model download, opt-in AI transmission. | APP `test_initial_page_has_privacy_note_and_no_data` | ✅ |
| F-04 | AI summary is opt-in: payload preview → consent tick bound to the exact payload → explicit button. Sends only the filtered selection, with pseudonyms and identifier masking on by default. Personality prompt replaced by a factual summary prompt. | APP `test_no_ai_request_without_opt_in_and_click`, `test_ai_respects_selected_participants`; AI `test_transcript_*` | ✅ |
| F-05 | Results keyed by SHA-256 of prompt version, system prompt, model, token limit and exact chunks; stored per session; never re-requested on reruns; per-session request budget. | APP `test_unrelated_reruns_do_not_repeat_paid_requests`, `test_request_budget_per_session`; AI `test_cache_key_depends_on_input_model_and_prompt` | ✅ |
| F-06 | Missing keys disable only the AI action; errors are mapped to user-safe messages; `client.showErrorDetails = "none"`; sections wrapped so one failure does not end the page; logs record exception types only. | APP `test_missing_credentials_keep_analytics_working`, `test_provider_timeout_shows_safe_error_and_keeps_dashboard`, `test_logs_do_not_contain_chat_content`; AI `test_missing_key` | ✅ |
| F-07 | 20 MB limit (widget + config + code), 300k-message and 2M-line limits, ZIP member/size guards (no trust in declared sizes), UTF-8/UTF-16 detection with replacement warnings, binary rejection. | P `test_read_upload_*`, `test_message_limit_keeps_most_recent`; APP `test_oversized_upload_is_rejected`, `test_upload_txt_and_zip_and_malformed` | ✅ |
| F-08 | Devcontainer runs plain `streamlit run app.py`; XSRF stays enabled (config). | Config review | ✅ |
| F-09 | `__pycache__/` ignored. History purge in runbook. | `git ls-files` | ✅ · 👤 history |

## Parsing

| ID | Fix | Tests | Status |
|---|---|---|---|
| F-10 | Record-based parser in `wa_analyzer/parsing.py` (module renamed from `preprcosseing.py`); every header line starts a record. | P `test_android_us_12h_preserves_first_message`, `test_single_message_chat` | ✅ |
| F-11 | Android and iOS styles; `/ . -` separators; 2/4-digit years; 12/24-hour; optional seconds; `am/pm/a.m.`; NBSP/NNBSP/BOM/bidi marks; DMY/MDY/YMD with ambiguity detection (impossible values, then chronological order, then locale hint) and an explicit choice in the UI. | P `test_dmy_24h_four_digit_year`, `test_dotted_dates_and_iso_dates`, `test_ios_bracketed_with_seconds_and_marks`, `test_android_narrow_nbsp_before_am_pm`, `test_lowercase_and_dotted_am_pm`, `test_bom_at_file_start_is_ignored`, `test_ambiguous_dates_flagged_and_explicit_choice_respected`, `test_ambiguity_resolved_by_chronological_order`; APP `test_ambiguous_dates_offer_explicit_choice` | ✅ |
| F-12 | Timestamp and content come from the same header match; continuation lines join the current record; line splitting ignores U+2028 etc. | P `test_timestamp_like_text_inside_message_keeps_alignment`, `test_multiline_and_colons_in_content_and_alignment`, `test_unicode_line_separator_inside_message_is_not_a_new_line` | ✅ |
| F-13 | Anchored English system-event patterns (never crossing a `:`), iOS U+200E system lines, media placeholders, deleted markers, edited markers; original `message` kept beside derived `text`. Invalid timestamps and preamble lines are reported with line numbers, never content. | P `test_system_media_deleted_edited_classification`, `test_phone_number_and_unicode_senders_and_emoji_preserved`, `test_invalid_dates_reported_not_invented`, `test_preamble_lines_reported`, `test_empty_or_unsupported_input_raises_safe_error`, `test_system_only_and_media_only_exports` | ✅ |
| F-14 | `day_name`/`period` columns removed; weekday derived where needed with real names. Stable chronological sort; naive timestamps (no invented timezone). | A `test_daily_hourly_weekday`; P `test_stable_chronological_sort_keeps_ties_in_file_order`, `test_no_timezone_is_invented` | ✅ |

## Statistics and interpretation

| ID | Fix | Tests | Status |
|---|---|---|---|
| F-15 | `overview()` reports active days, calendar span and both averages on the filtered selection. | A `test_overview_hand_calculated`; E `test_overview_table_values_match_overview` | ✅ |
| F-16 | One `Filters` object drives every tab, download and AI input; timing metrics take the full chat plus filters. | A `test_filters_apply_consistently`; APP `test_participant_filter_applies_to_metrics`, `test_ai_respects_selected_participants` | ✅ |
| F-17 | Response-gap estimates (sender changes within a session) and same-sender follow-up gaps computed on the full conversation, then filtered; labelled as estimates. | A `test_response_gaps_full_conversation_then_filter` | ✅ |
| F-18 | Silence gaps computed from the complete timeline, independent of sentiment, with neutral labels (`last_sender_before`, `first_sender_after`). | A `test_silence_gaps_from_complete_timeline` | ✅ |
| F-19 | Removed initiation/effort, emotional-imbalance, keyword-topic and relationship claims; added documented idle-gap sessions and "session starter" with a timing-only caveat. | A `test_sessions_idle_rule`; UI text review | ✅ |
| F-20 | Year-aware, zero-filled daily/weekly/monthly series; Monday-first weekdays. | A `test_monthly_and_weekly_are_year_aware_and_chronological`, `test_daily_hourly_weekday` | ✅ |
| F-21 | Emoji counted as full sequences via `emoji.emoji_list`. | A `test_words_and_emoji`; E `test_text_helpers` | ✅ |
| F-22 | Helpers are pure and return the same types for empty input. | A `test_functions_do_not_mutate_input`, `test_empty_inputs_return_consistent_types`, `test_system_only_chat` | ✅ |
| F-23 | Empty-state messages instead of charts; word cloud built only from non-empty frequencies. | A `test_empty_inputs_return_consistent_types`; APP `test_participant_filter_applies_to_metrics` | ✅ |
| F-24 | Six focused tabs; a real weekday × hour heatmap; bounded charts (top 15) and tables. | BR `test_desktop_flow_heatmap_and_clear` (asserts a Plotly heatmap layer is drawn) | ✅ |
| F-25 | Derived state keyed by upload SHA-256 + date order + filters; a new upload clears memo, sentiment and AI results. | APP `test_upload_replacement_invalidates_results_even_with_same_name_and_people` | ✅ |

## Sentiment

| ID | Fix | Tests | Status |
|---|---|---|---|
| F-26 | Probabilities per class kept; label = argmax; confidence shown separately; optional polarity index = P(pos) − P(neg) documented as uncalibrated; charts show class shares. | S `test_confident_negative_is_negative_not_positive`, `test_normalize_label_map_variants`, `test_aggregations` | ✅ |
| F-27 | Model-card preprocessing (`@user`, `http`), emoji kept, text length bounded. | S `test_preprocess_follows_model_card_and_keeps_emoji` | ✅ |
| F-28 | Lazy import, explicit button, cached model weights only, batches of 32, 128 tokens, 3,000-message cap, per-item retry, failures marked per row; load failures shown without stopping the page. | S `test_row_alignment_with_skips_and_failures`, `test_message_cap_keeps_most_recent`, `test_empty_selection`; APP `test_sentiment_runs_only_on_click_and_is_reused`, `test_sentiment_model_failure_is_contained`, `test_sentiment_unavailable_state` | ✅ (mocked) · 🟡 real model not runnable here (RM) |
| F-29 | Limitations documented in the UI and README; no accuracy claims. | Text review | ✅ |

## External AI

| ID | Fix | Tests | Status |
|---|---|---|---|
| F-30 | Configurable `WA_TOGETHER_MODELS` (defaults `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, from secondary sources); unavailable-model errors explained. | AI `test_non_retryable_errors` (404 / model 400) | 🟡 model IDs not verifiable against official docs from this environment |
| F-31 | System prompt separated from delimited data; delimiter spoofing removed; no tools; output escaped, links removed, only known headings/bullets re-created; timeouts, deadline, bounded retries, stream-interruption handling, empty-output handling. | AI `test_render_safe_markdown_neutralises_html_images_and_links`, `test_streaming_success_and_request_shape`, `test_rate_limit_*`, `test_timeout_and_connection_errors_are_retried_then_reported`, `test_interrupted_stream_is_not_retried_and_keeps_partial`, `test_stream_without_done_is_incomplete`, `test_empty_output_reported`, `test_deadline_stops_request`, `test_summarize_*`; APP `test_model_output_is_rendered_safely`, `test_shared_key_present_but_not_enabled_is_not_used` | ✅ (mocked provider) |

## Engineering, dependencies and delivery

| ID | Fix | Tests / verification | Status |
|---|---|---|---|
| F-32 | NLTK removed; built-in English stop-word list. | Import has no network side effects (offline test run) | ✅ |
| F-33 | `requirements.txt` (core, 6 pins), `requirements-ml.txt` (optional CPU extras with safe lower bounds), `requirements-dev.txt`. | `pip check`, `uv pip compile` on 3.11/3.13, `pip-audit` (VALIDATION_REPORT §3) | ✅ |
| F-34 | `wa_analyzer` package (parsing, analytics, text, sentiment, ai_summary, export, config, sample, ui); `app.py` is 7 lines; no user content in `unsafe_allow_html`; header no longer hidden. | Ruff lint/format; BR `test_mobile_layout_has_no_horizontal_scroll` | ✅ |
| F-35 | `ci.yml` (lint, tests on 3.11/3.13, audit, browser), `real-model.yml` (manual); Azure workflow runs tests and packages `.streamlit/config.toml`. | YAML validated; zip command checked locally | ✅ · 🟡 workflows not executed on GitHub from here |
| F-36 | README rewritten to verified features only. | Review | ✅ |
| — | Performance: features computed once per upload; core statistics on 300k messages 168 s → 3.2 s. | `scripts/benchmark.py` (VALIDATION_REPORT §4) | ✅ |
| — | CSV downloads protected against formula injection; downloads exclude message text by default. | E `test_csv_safe`, `test_to_safe_csv_round_trip_neutralises_names`, `test_export_zip_and_report_exclude_message_text`, `test_pseudonym_map` | ✅ |
