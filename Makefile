.PHONY: setup run test clean

# Create venv and install all dependencies (app + dev + Playwright browser)
setup:
	python -m venv venv
	venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt
	venv\Scripts\playwright install chromium

# Start the live transcription server (opens browser automatically)
run:
	venv\Scripts\python run.py

# Run the E2E test suite with the tiny model
test:
	set WHISPER_MODEL=tiny && venv\Scripts\python -m pytest tests/e2e/test_app.py -v

# Remove venv and caches
clean:
	if exist venv rmdir /s /q venv
	if exist .pytest_cache rmdir /s /q .pytest_cache
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
