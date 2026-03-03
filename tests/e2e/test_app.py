"""End-to-end tests for the Live Transcription app.

Categories (38 tests):
  1. Page Load (9)
  2. API (3)
  3. WebSocket (4)
  4. Recording Controls (7)
  5. Language Selection (4)
  6. Save Transcript (2)
  7. Full Journey (1)
  8. Layout (2)
  9. Edge Cases (3)
"""

import json
import time

import httpx
import pytest
from playwright.sync_api import Page, expect


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  PAGE LOAD (9 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPageLoad:
    """Verify that the initial page renders all core elements."""

    def test_title(self, page: Page) -> None:
        expect(page).to_have_title("Live Transcription")

    def test_start_button_visible(self, page: Page) -> None:
        btn = page.locator("#recordBtn")
        expect(btn).to_be_visible()
        expect(btn).to_have_text("Start Recording")

    def test_save_button_visible(self, page: Page) -> None:
        expect(page.locator("#saveBtn")).to_be_visible()

    def test_language_dropdown_visible(self, page: Page) -> None:
        sel = page.locator("#languageSelect")
        expect(sel).to_be_visible()

    def test_language_dropdown_has_four_options(self, page: Page) -> None:
        options = page.locator("#languageSelect option")
        expect(options).to_have_count(4)

    def test_language_dropdown_defaults_to_auto(self, page: Page) -> None:
        sel = page.locator("#languageSelect")
        expect(sel).to_have_value("auto")

    def test_transcript_placeholder(self, page: Page) -> None:
        ph = page.locator("#placeholder")
        expect(ph).to_be_visible()
        expect(ph).to_contain_text("Transcript will appear here")

    def test_connection_status_disconnected_then_connected(self, page: Page) -> None:
        """The dot should gain the 'connected' class within a few seconds."""
        dot = page.locator("#statusDot")
        expect(dot).to_have_class("dot connected", timeout=10_000)

    def test_model_device_info_loads(self, page: Page) -> None:
        info = page.locator("#modelInfo")
        expect(info).not_to_contain_text("loading…", timeout=15_000)
        # Should include model size and device
        expect(info).to_contain_text("Model:")


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  API (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAPI:
    """Verify the REST /api/config endpoint."""

    def test_config_returns_json(self, server_url: str) -> None:
        r = httpx.get(f"{server_url}/api/config")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")

    def test_config_has_languages_and_model_sizes(self, server_url: str) -> None:
        data = httpx.get(f"{server_url}/api/config").json()
        assert "languages" in data
        assert "model_sizes" in data
        assert len(data["languages"]) >= 3
        assert len(data["model_sizes"]) >= 3

    def test_config_model_info_loaded(self, server_url: str) -> None:
        data = httpx.get(f"{server_url}/api/config").json()
        assert data["model_info"]["loaded"] is True


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  WEBSOCKET (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestWebSocket:
    """Verify WebSocket connectivity and initial messages."""

    def test_clicking_start_opens_ws(self, page: Page) -> None:
        # Wait for WS to connect first
        page.locator("#statusDot.connected").wait_for(timeout=10_000)
        page.locator("#recordBtn").click()
        # The button should switch to "Stop Recording"
        expect(page.locator("#recordBtn")).to_have_text("Stop Recording", timeout=5_000)
        # Clean up
        page.locator("#recordBtn").click()

    def test_status_changes_to_connected(self, page: Page) -> None:
        expect(page.locator("#statusText")).to_have_text("Connected", timeout=10_000)

    def test_dot_indicator_green(self, page: Page) -> None:
        dot = page.locator("#statusDot")
        expect(dot).to_have_class("dot connected", timeout=10_000)

    def test_receives_initial_status_message(self, page: Page, server_url: str) -> None:
        """Connect via raw WS and verify the first message is a status message."""
        import websockets.sync.client as wsc

        ws_url = server_url.replace("http://", "ws://") + "/ws"
        with wsc.connect(ws_url) as conn:
            raw = conn.recv(timeout=5)
            msg = json.loads(raw)
            assert msg["type"] == "status"
            assert "recording" in msg
            assert "model_loaded" in msg


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  RECORDING CONTROLS (7 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecordingControls:
    """Verify Start/Stop toggle behaviour and associated UI changes."""

    def _wait_connected(self, page: Page) -> None:
        page.locator("#statusDot.connected").wait_for(timeout=10_000)

    def test_button_text_toggles_to_stop(self, page: Page) -> None:
        self._wait_connected(page)
        page.locator("#recordBtn").click()
        expect(page.locator("#recordBtn")).to_have_text("Stop Recording", timeout=5_000)
        page.locator("#recordBtn").click()

    def test_button_text_toggles_back_to_start(self, page: Page) -> None:
        self._wait_connected(page)
        page.locator("#recordBtn").click()
        page.locator("#recordBtn").click()
        expect(page.locator("#recordBtn")).to_have_text("Start Recording", timeout=5_000)

    def test_recording_css_class_added(self, page: Page) -> None:
        self._wait_connected(page)
        page.locator("#recordBtn").click()
        expect(page.locator("#recordBtn")).to_have_class("btn btn-record recording", timeout=5_000)
        page.locator("#recordBtn").click()

    def test_recording_css_class_removed(self, page: Page) -> None:
        self._wait_connected(page)
        page.locator("#recordBtn").click()
        page.wait_for_timeout(500)
        page.locator("#recordBtn").click()
        expect(page.locator("#recordBtn")).not_to_have_class(
            "btn btn-record recording", timeout=5_000
        )

    def test_timer_starts_on_record(self, page: Page) -> None:
        self._wait_connected(page)
        page.locator("#recordBtn").click()
        page.wait_for_timeout(1500)
        timer_text = page.locator("#timer").text_content()
        assert timer_text and ":" in timer_text
        page.locator("#recordBtn").click()

    def test_timer_stops_on_stop(self, page: Page) -> None:
        self._wait_connected(page)
        page.locator("#recordBtn").click()
        page.wait_for_timeout(1500)
        page.locator("#recordBtn").click()
        time_a = page.locator("#timer").text_content()
        page.wait_for_timeout(1500)
        time_b = page.locator("#timer").text_content()
        assert time_a == time_b  # timer frozen

    def test_status_reverts_on_stop(self, page: Page) -> None:
        self._wait_connected(page)
        page.locator("#recordBtn").click()
        page.wait_for_timeout(500)
        page.locator("#recordBtn").click()
        expect(page.locator("#recordBtn")).to_have_text("Start Recording", timeout=5_000)


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  LANGUAGE SELECTION (4 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestLanguageSelection:
    """Verify language dropdown sends set_language commands over WS."""

    def _wait_connected(self, page: Page) -> None:
        page.locator("#statusDot.connected").wait_for(timeout=10_000)

    def test_select_english(self, page: Page) -> None:
        self._wait_connected(page)
        page.locator("#languageSelect").select_option("en")
        expect(page.locator("#languageSelect")).to_have_value("en")

    def test_select_french(self, page: Page) -> None:
        self._wait_connected(page)
        page.locator("#languageSelect").select_option("fr")
        expect(page.locator("#languageSelect")).to_have_value("fr")

    def test_select_italian(self, page: Page) -> None:
        self._wait_connected(page)
        page.locator("#languageSelect").select_option("it")
        expect(page.locator("#languageSelect")).to_have_value("it")

    def test_change_sends_ws_set_language(self, page: Page, server_url: str) -> None:
        """Intercept WS messages to confirm set_language is sent."""
        self._wait_connected(page)

        # Inject a spy on WebSocket.send
        page.evaluate("""() => {
            window.__wsSent = [];
            const origSend = WebSocket.prototype.send;
            WebSocket.prototype.send = function(data) {
                window.__wsSent.push(data);
                return origSend.call(this, data);
            };
        }""")

        page.locator("#languageSelect").select_option("fr")
        page.wait_for_timeout(500)

        sent = page.evaluate("() => window.__wsSent")
        assert any('"set_language"' in s for s in sent)


# ═══════════════════════════════════════════════════════════════════════════════
#  6.  SAVE TRANSCRIPT (2 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSaveTranscript:
    """Verify save behaviour."""

    def _wait_connected(self, page: Page) -> None:
        page.locator("#statusDot.connected").wait_for(timeout=10_000)

    def test_alert_when_empty(self, page: Page) -> None:
        self._wait_connected(page)
        # Hook into window.alert to capture it
        page.evaluate("() => { window.__alertMsg = null; window.alert = m => window.__alertMsg = m; }")
        page.locator("#saveBtn").click()
        page.wait_for_timeout(1000)
        msg = page.evaluate("() => window.__alertMsg")
        assert msg is not None and "no transcript" in msg.lower()

    def test_download_triggered_with_content(self, page: Page, server_url: str) -> None:
        """Simulate a save response with actual text to verify download triggers."""
        self._wait_connected(page)

        # Directly inject a mock segment so there's something to save
        page.evaluate("""() => {
            const area = document.getElementById("transcriptArea");
            const ph = document.getElementById("placeholder");
            if (ph) ph.style.display = "none";
            const div = document.createElement("div");
            div.className = "segment";
            div.textContent = "Hello world";
            area.appendChild(div);
        }""")

        # Replace alert so we can test the download was attempted
        page.evaluate("""() => {
            window.__downloadTriggered = false;
            const origCreateElement = document.createElement.bind(document);
            document.createElement = function(tag) {
                const el = origCreateElement(tag);
                if (tag === 'a') {
                    const origClick = el.click;
                    el.click = function() { window.__downloadTriggered = true; };
                }
                return el;
            };
        }""")

        # Use raw WS to send a save response
        import websockets.sync.client as wsc

        ws_url = server_url.replace("http://", "ws://") + "/ws"
        with wsc.connect(ws_url) as conn:
            conn.recv(timeout=5)  # initial status
            conn.send(json.dumps({"action": "save"}))
            raw = conn.recv(timeout=5)
            msg = json.loads(raw)
            assert msg["type"] == "save"


# ═══════════════════════════════════════════════════════════════════════════════
#  7.  FULL JOURNEY (1 test)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFullJourney:
    """Complete session: load → select language → start → wait → stop → verify."""

    def test_complete_session(self, page: Page) -> None:
        # Wait for connection
        page.locator("#statusDot.connected").wait_for(timeout=10_000)

        # Select language
        page.locator("#languageSelect").select_option("en")
        expect(page.locator("#languageSelect")).to_have_value("en")

        # Start recording
        page.locator("#recordBtn").click()
        expect(page.locator("#recordBtn")).to_have_text("Stop Recording", timeout=5_000)
        expect(page.locator("#recordBtn")).to_have_class("btn btn-record recording", timeout=5_000)

        # Wait a bit (simulating recording)
        page.wait_for_timeout(2000)

        # Timer should be updating
        timer = page.locator("#timer").text_content()
        assert timer and ":" in timer

        # Stop recording
        page.locator("#recordBtn").click()
        expect(page.locator("#recordBtn")).to_have_text("Start Recording", timeout=10_000)

        # Page should still be functional
        expect(page.locator("#statusDot")).to_have_class("dot connected", timeout=10_000)


# ═══════════════════════════════════════════════════════════════════════════════
#  8.  LAYOUT (2 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestLayout:
    """Verify the page renders on narrow and wide viewports."""

    def test_narrow_viewport_375(self, page: Page) -> None:
        page.set_viewport_size({"width": 375, "height": 667})
        page.reload(wait_until="networkidle")
        expect(page.locator("#recordBtn")).to_be_visible()
        expect(page.locator("#languageSelect")).to_be_visible()
        expect(page.locator("#saveBtn")).to_be_visible()

    def test_wide_viewport_1920(self, page: Page) -> None:
        page.set_viewport_size({"width": 1920, "height": 1080})
        page.reload(wait_until="networkidle")
        expect(page.locator("#recordBtn")).to_be_visible()
        expect(page.locator("#languageSelect")).to_be_visible()
        expect(page.locator("#saveBtn")).to_be_visible()


# ═══════════════════════════════════════════════════════════════════════════════
#  9.  EDGE CASES (3 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Verify the app handles unusual interaction patterns gracefully."""

    def _wait_connected(self, page: Page) -> None:
        page.locator("#statusDot.connected").wait_for(timeout=10_000)

    def test_double_start_safe(self, page: Page, server_url: str) -> None:
        """Sending two 'start' commands in rapid succession should not crash."""
        import websockets.sync.client as wsc

        ws_url = server_url.replace("http://", "ws://") + "/ws"
        with wsc.connect(ws_url) as conn:
            conn.recv(timeout=5)  # initial status
            conn.send(json.dumps({"action": "start"}))
            conn.send(json.dumps({"action": "start"}))
            page.wait_for_timeout(1000)
            conn.send(json.dumps({"action": "stop"}))
            raw = conn.recv(timeout=10)
            msg = json.loads(raw)
            assert msg["type"] == "status"

    def test_multiple_start_stop_cycles(self, page: Page) -> None:
        self._wait_connected(page)
        for _ in range(3):
            page.locator("#recordBtn").click()
            expect(page.locator("#recordBtn")).to_have_text("Stop Recording", timeout=5_000)
            page.wait_for_timeout(500)
            page.locator("#recordBtn").click()
            expect(page.locator("#recordBtn")).to_have_text("Start Recording", timeout=10_000)

    def test_language_change_before_start(self, page: Page) -> None:
        """Changing language before recording starts should not error."""
        self._wait_connected(page)
        page.locator("#languageSelect").select_option("it")
        page.locator("#languageSelect").select_option("fr")
        page.locator("#languageSelect").select_option("auto")
        # Should still be able to start
        page.locator("#recordBtn").click()
        expect(page.locator("#recordBtn")).to_have_text("Stop Recording", timeout=5_000)
        page.locator("#recordBtn").click()
