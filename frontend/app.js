/* Live Transcription — WebSocket client & DOM controller */

(function () {
  "use strict";

  // ── DOM refs ────────────────────────────────────────────────────────────
  const recordBtn = document.getElementById("recordBtn");
  const languageSelect = document.getElementById("languageSelect");
  const saveBtn = document.getElementById("saveBtn");
  const transcriptArea = document.getElementById("transcriptArea");
  const placeholder = document.getElementById("placeholder");
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const modelInfo = document.getElementById("modelInfo");
  const timerEl = document.getElementById("timer");

  // ── State ───────────────────────────────────────────────────────────────
  let ws = null;
  let recording = false;
  let timerHandle = null;
  let startTime = null;

  // ── Helpers ─────────────────────────────────────────────────────────────

  function setConnected(connected) {
    if (connected) {
      statusDot.classList.add("connected");
      statusText.textContent = "Connected";
      recordBtn.disabled = false;
    } else {
      statusDot.classList.remove("connected");
      statusText.textContent = "Disconnected";
      recordBtn.disabled = true;
    }
  }

  function formatTime(seconds) {
    const m = Math.floor(seconds / 60)
      .toString()
      .padStart(2, "0");
    const s = Math.floor(seconds % 60)
      .toString()
      .padStart(2, "0");
    return m + ":" + s;
  }

  function startTimer() {
    startTime = Date.now();
    timerHandle = setInterval(function () {
      const elapsed = (Date.now() - startTime) / 1000;
      timerEl.textContent = formatTime(elapsed);
    }, 500);
  }

  function stopTimer() {
    clearInterval(timerHandle);
    timerHandle = null;
  }

  function appendSegment(text, language, isPartial) {
    // Hide placeholder
    if (placeholder) placeholder.style.display = "none";

    const div = document.createElement("div");
    div.className = "segment" + (isPartial ? " partial" : "");

    const badge = document.createElement("span");
    badge.className = "lang-badge";
    badge.textContent = language || "??";

    const span = document.createElement("span");
    span.textContent = text;

    div.appendChild(badge);
    div.appendChild(span);
    transcriptArea.appendChild(div);

    // Auto-scroll
    transcriptArea.scrollTop = transcriptArea.scrollHeight;
  }

  function sendWS(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(obj));
    }
  }

  // ── WebSocket ───────────────────────────────────────────────────────────

  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(proto + "//" + location.host + "/ws");

    ws.addEventListener("open", function () {
      setConnected(true);
    });

    ws.addEventListener("close", function () {
      setConnected(false);
      // Attempt reconnect after a delay
      setTimeout(connect, 2000);
    });

    ws.addEventListener("error", function () {
      ws.close();
    });

    ws.addEventListener("message", function (event) {
      var msg;
      try {
        msg = JSON.parse(event.data);
      } catch (_) {
        return;
      }

      if (msg.type === "status") {
        recording = msg.recording;
        recordBtn.textContent = recording
          ? "Stop Recording"
          : "Start Recording";
        if (recording) {
          recordBtn.classList.add("recording");
        } else {
          recordBtn.classList.remove("recording");
        }
      } else if (msg.type === "transcript") {
        appendSegment(msg.text, msg.language, msg.is_partial);
      } else if (msg.type === "save") {
        if (!msg.text || msg.text.trim() === "") {
          alert("No transcript to save.");
          return;
        }
        // Trigger download
        var blob = new Blob([msg.text], { type: "text/plain" });
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = "transcript.txt";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    });
  }

  // ── Event listeners ─────────────────────────────────────────────────────

  recordBtn.addEventListener("click", function () {
    if (!recording) {
      sendWS({ action: "start" });
      startTimer();
    } else {
      sendWS({ action: "stop" });
      stopTimer();
    }
  });

  languageSelect.addEventListener("change", function () {
    sendWS({ action: "set_language", language: languageSelect.value });
  });

  saveBtn.addEventListener("click", function () {
    sendWS({ action: "save" });
  });

  // ── Load config & initial connect ───────────────────────────────────────

  fetch("/api/config")
    .then(function (res) {
      return res.json();
    })
    .then(function (cfg) {
      var mi = cfg.model_info;
      modelInfo.textContent =
        "Model: " +
        mi.size +
        " | Device: " +
        mi.device +
        " (" +
        mi.compute_type +
        ")" +
        (mi.loaded ? "" : " [loading…]");
    })
    .catch(function () {
      modelInfo.textContent = "Could not load config";
    });

  connect();
})();
