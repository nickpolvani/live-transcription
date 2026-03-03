# Setup & Run Guide – Live Transcription

Step-by-step instructions for setting up the environment and running the app on **Windows**.
Linux/macOS users: swap the activate command as noted below.

---

## 1. Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.9 or newer (3.11 recommended). Download from <https://www.python.org/downloads/> |
| **Microphone** | Any working mic recognised by Windows |
| **GPU (optional)** | NVIDIA GPU with CUDA 11.8+ drivers for hardware acceleration. Without it the app runs on CPU — slower but fully functional |
| **Disk space** | ~150 MB for dependencies + model size (see table below) |

### Model disk / RAM usage (downloaded automatically on first run)

| Model | Download | RAM (CPU) | VRAM (GPU) |
|---|---|---|---|
| tiny | ~75 MB | ~400 MB | ~1 GB |
| base | ~150 MB | ~600 MB | ~1 GB |
| small | ~500 MB | ~1 GB | ~2 GB |
| **medium** (default) | ~1.5 GB | ~2.1 GB | ~5 GB |
| large-v3 | ~3 GB | ~4 GB | ~10 GB |

---

## 2. Clone the repository

```bash
git clone https://github.com/YOUR_USER/live-transcription.git
cd live-transcription
```

---

## 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```powershell
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (cmd)
venv\Scripts\activate.bat

# Linux / macOS
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

- `faster-whisper` — Whisper speech recognition (CTranslate2 backend)
- `ctranslate2` — inference engine
- `onnxruntime` — Silero VAD runtime
- `fastapi` + `uvicorn` — web server
- `sounddevice` — microphone capture
- `numpy`, `websockets`

### GPU users only

If you have an NVIDIA GPU and want CUDA acceleration, also install the CUDA-enabled
CTranslate2 wheel. Refer to
<https://opennmt.net/CTranslate2/installation.html> for CUDA-specific wheels.
The app auto-detects GPU vs CPU at startup — no code changes needed.

---

## 5. Run the app

```bash
python run.py
```

What happens:

1. The Whisper model downloads on first run (this is a one-time step — cached in
   `~/.cache/huggingface/`).
2. The model loads into memory.
3. A web server starts on **http://localhost:8080**.
4. Your default browser opens automatically.

### Choose a different model

Set the `WHISPER_MODEL` environment variable **before** launching:

```powershell
# PowerShell — use the tiny model (fastest, least accurate)
$env:WHISPER_MODEL = "tiny"
python run.py

# cmd
set WHISPER_MODEL=tiny
python run.py

# Linux / macOS
WHISPER_MODEL=tiny python run.py
```

Valid values: `tiny`, `base`, `small`, `medium`, `large-v3`.

---

## 6. Using the UI

| Action | How |
|---|---|
| **Start recording** | Click the red **Start Recording** button |
| **Stop recording** | Click the button again (now reads **Stop Recording**) |
| **Change language** | Pick from the dropdown: Auto-detect, English, French, Italian |
| **Save transcript** | Click **Save Transcript** — a `.txt` file downloads |

- Transcribed text appears in near-real-time (~2-3 s latency).
- The footer shows the active model, device (CPU/CUDA), and a recording timer.
- The green dot in the header confirms the WebSocket connection is live.

---

## 7. Running the E2E tests

Install dev dependencies:

```bash
pip install -r requirements-dev.txt
playwright install chromium
```

Run the test suite (uses the `tiny` model automatically):

```powershell
# PowerShell
$env:WHISPER_MODEL = "tiny"
python -m pytest tests/e2e/test_app.py -v
```

```bash
# Linux / macOS / cmd
WHISPER_MODEL=tiny python -m pytest tests/e2e/test_app.py -v
```

All 35 tests should pass in under 60 seconds.

---

## 8. Configuration reference

All defaults live in `backend/config.py`. Override via env vars or by editing the file directly.

| Setting | Default | Env var | Description |
|---|---|---|---|
| `model_size` | `medium` | `WHISPER_MODEL` | Whisper model variant |
| `device` | auto-detected | — | `cuda` if NVIDIA GPU available, else `cpu` |
| `compute_type` | auto (`float16` GPU / `int8` CPU) | — | CTranslate2 quantisation |
| `language` | `auto` | — | `auto`, `en`, `fr`, or `it` |
| `sample_rate` | `16000` | — | Audio sample rate in Hz |
| `chunk_duration` | `2.0` | — | Seconds of audio per transcription chunk |
| `vad_threshold` | `0.5` | — | Silero VAD speech probability threshold |
| `host` | `127.0.0.1` | — | Server bind address |
| `port` | `8080` | — | Server port |

---

## 9. Troubleshooting

| Problem | Solution |
|---|---|
| **`ModuleNotFoundError`** | Make sure the venv is activated (`(venv)` in prompt) |
| **Model download stalls** | Check internet; HuggingFace Hub caches to `~/.cache/huggingface/` |
| **No microphone input** | Verify your mic is set as default in Windows Sound Settings |
| **`onnxruntime` DLL error** | Run `pip install onnxruntime==1.20.1` (newer versions may have DLL issues on Windows) |
| **`ctranslate2` access violation** | Run `pip install "ctranslate2>=4.4.0,<4.6.0"` (v4.7+ crashes on some Windows configs) |
| **Port 8080 in use** | Edit `port` in `backend/config.py` or kill the conflicting process |
| **CUDA not detected** | Ensure NVIDIA drivers + CUDA toolkit are installed; run `nvidia-smi` to verify |
