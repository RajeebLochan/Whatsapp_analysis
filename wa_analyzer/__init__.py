"""WhatsApp Chat Analyzer: parsing, analytics and optional AI features.

Modules
-------
- ``parsing``     – WhatsApp export parser (Android/iOS formats).
- ``analytics``   – Pure, non-mutating statistics on parsed chats.
- ``text``        – Word, link and emoji extraction helpers.
- ``sentiment``   – Optional, lazily loaded sentiment model.
- ``ai_summary``  – Optional, opt-in Together AI summaries.
- ``export``      – Aggregate CSV / report downloads with formula protection.
- ``config``      – Limits and secret loading.
- ``sample``      – Synthetic sample chat for demos and tests.
- ``ui``          – Streamlit user interface.
"""

__version__ = "2.0.0"
