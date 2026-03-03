# Live Transcription

Real-time speech-to-text transcription powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with a lightweight web UI.

## Features

- **Real-time transcription** — audio is captured from your microphone and transcribed in ~2-3 second chunks
- **Multilingual** — supports English, French, Italian, and auto-detection
- **Voice Activity Detection** — Silero VAD skips silence to save compute
- **GPU acceleration** — automatically uses NVIDIA CUDA when available, falls back to CPU
- **Web UI** — clean dark-themed single-page app served over WebSocket
- **Save transcripts** — download the full session as a `.txt` file

## Model Sizes

| Model     | Parameters | English-only | Multilingual | VRAM (approx.) |
|-----------|-----------|:------------:|:------------:|:--------------:|
| tiny      | 39 M      | ✓            | ✓            | ~1 GB          |
| base      | 74 M      | ✓            | ✓            | ~1 GB          |
| small     | 244 M     | ✓            | ✓            | ~2 GB          |
| **medium**| 769 M     | ✓            | ✓            | ~5 GB          |
| large-v3  | 1550 M    | ✓            | ✓            | ~10 GB         |

The default model is **medium** — best accuracy/speed tradeoff for multilingual use.

## Requirements

- Python 3.9+
- A working microphone
- (Optional) NVIDIA GPU with CUDA for hardware acceleration

## Quick Start

```bash
git clone https://github.com/nickpolvani/live-transcription.git
cd live-transcription
make setup   # creates venv, installs deps + Playwright
make run     # starts server and opens browser
```

### Manual installation (without Make)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
python run.py
```

### For development / testing

```bash
# Already included in `make setup`, or manually:
pip install -r requirements-dev.txt
playwright install chromium
```

## Usage

```bash
make run
# or: python run.py
```

The browser will open automatically at **http://localhost:8080**.

1. Click **Start Recording** to begin capturing audio from your microphone.
2. Select a language from the dropdown or leave it on **Auto-detect**.
3. Transcription segments appear in near-real-time.
4. Click **Stop Recording** when finished.
5. Click **Save Transcript** to download the full text as a `.txt` file.

## Configuration

Edit `backend/config.py` to change defaults:

| Setting         | Default  | Description                              |
|-----------------|----------|------------------------------------------|
| `model_size`    | medium   | Whisper model variant                    |
| `device`        | auto     | `cuda` if available, else `cpu`          |
| `language`      | auto     | ISO language code or `auto`              |
| `chunk_duration`| 2.0      | Seconds of audio per transcription chunk |
| `vad_threshold` | 0.5      | Silero VAD speech probability threshold  |
| `port`          | 8080     | HTTP server port                         |

## Running Tests

```bash
make test
# or: WHISPER_MODEL=tiny python -m pytest tests/e2e/test_app.py -v
```

## Project Structure

```
live-transcription/
├── backend/
│   ├── __init__.py
│   ├── config.py          # Settings dataclass, device detection
│   ├── transcriber.py     # faster-whisper wrapper
│   ├── audio_capture.py   # sounddevice mic capture + ring buffer
│   └── server.py          # FastAPI app, WebSocket endpoint
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/
│   ├── e2e/
│   │   ├── conftest.py
│   │   └── test_app.py
│   └── fixtures/
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── Makefile
├── run.py
└── README.md
```

## License

MIT
