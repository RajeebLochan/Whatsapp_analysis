"""Real-browser smoke test (Playwright + Chromium) for behaviour AppTest cannot see:
actual rendering of charts, tabs, the heatmap, mobile layout and the clear action.

Skipped unless the ``playwright`` package and a Chromium build are available.
Set ``PLAYWRIGHT_CHROMIUM_EXECUTABLE`` to use a pre-installed browser.
Screenshots are written to tests/browser/artifacts/ (git-ignored).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.browser
sync_api = pytest.importorskip("playwright.sync_api")

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    env = {**os.environ, "STREAMLIT_SERVER_HEADLESS": "true"}
    for var in ("TOGETHER_API_KEY", "WA_ENABLE_SHARED_AI_KEY"):
        env.pop(var, None)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            str(port),
            "--server.address",
            "127.0.0.1",
        ],
        cwd=REPO,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url + "/_stcore/health", timeout=2)  # noqa: S310 - local test server
            break
        except OSError:
            time.sleep(0.5)
    else:
        proc.terminate()
        pytest.fail("Streamlit server did not start")
    yield url
    proc.terminate()
    proc.wait(timeout=20)


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        exe = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
        try:
            b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Chromium not available: {type(exc).__name__}")
        yield b
        b.close()


def _load_sample(page, url):
    page.goto(url, wait_until="networkidle")
    page.get_by_role("button", name="Try the synthetic sample chat").click()
    page.get_by_text("Records parsed").first.wait_for(timeout=30_000)


def test_desktop_flow_heatmap_and_clear(server, browser):
    ARTIFACTS.mkdir(exist_ok=True)
    page = browser.new_page(viewport={"width": 1366, "height": 900}, device_scale_factor=1)
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    _load_sample(page, server)
    page.get_by_text("Detected format").first.wait_for()
    page.wait_for_selector(".js-plotly-plot", timeout=30_000)
    page.screenshot(path=str(ARTIFACTS / "desktop-overview.png"), full_page=True)

    page.get_by_role("tab", name="Activity").click()
    page.wait_for_selector(".js-plotly-plot .heatmaplayer", timeout=30_000)  # a real heatmap trace is drawn
    heatmap = page.locator(".js-plotly-plot").filter(has=page.locator(".heatmaplayer")).first
    heatmap.scroll_into_view_if_needed()
    heatmap.screenshot(path=str(ARTIFACTS / "desktop-heatmap.png"))

    page.get_by_role("tab", name="AI summary (optional)").click()
    page.get_by_text("Review exactly what will be sent").wait_for()
    assert page.get_by_role("button", name="Generate AI summary").is_disabled()

    page.get_by_role("button", name="Clear chat and results").click()
    page.get_by_text("Cleared. The chat and all results were removed").wait_for(timeout=30_000)
    assert page.get_by_text("Records parsed").count() == 0
    assert not errors, errors
    page.close()


def test_mobile_layout_has_no_horizontal_scroll(server, browser):
    ARTIFACTS.mkdir(exist_ok=True)
    page = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1, is_mobile=True)
    _load_sample(page, server)
    page.wait_for_selector(".js-plotly-plot", timeout=30_000)
    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    assert overflow <= 1, f"horizontal overflow of {overflow}px"
    tab = page.get_by_role("tab", name="Download")
    tab.scroll_into_view_if_needed()
    tab.click()
    page.get_by_role("button", name="Download report (.md)").wait_for()
    page.screenshot(path=str(ARTIFACTS / "mobile.png"), full_page=True)
    page.close()
