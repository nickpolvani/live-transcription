.PHONY: setup setup-gpu run run-gpu test test-latency clean

# Default model — override with: make run WHISPER_MODEL=large-v3
WHISPER_MODEL ?= medium

# Create venv and install all dependencies (app + dev + Playwright browser)
setup:
	python -m venv venv
	venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt
	venv\Scripts\playwright install chromium

# Install cuDNN for GPU inference (requires CUDA toolkit already on system)
setup-gpu: setup
	venv\Scripts\pip install nvidia-cudnn-cu12

# Start the live transcription server on CPU
run:
	set "WHISPER_MODEL=$(WHISPER_MODEL)" && venv\Scripts\python run.py

# Start the live transcription server on GPU (float16)
run-gpu:
	set "WHISPER_MODEL=$(WHISPER_MODEL)" && venv\Scripts\python run.py

# Run the E2E test suite with the tiny model
test:
	set WHISPER_MODEL=tiny && venv\Scripts\python -m pytest tests/e2e/test_app.py -v

# Run latency benchmarks
test-latency:
	set WHISPER_MODEL=tiny && venv\Scripts\python -m pytest tests/test_latency.py -v -s

# Remove venv and caches
clean:
	if exist venv rmdir /s /q venv
	if exist .pytest_cache rmdir /s /q .pytest_cache
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
