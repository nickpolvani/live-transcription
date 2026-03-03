"""Fixtures for end-to-end tests — starts the FastAPI server and provides Playwright page."""

import os
import subprocess
import sys
import time
from typing import Generator

import pytest
from playwright.sync_api import Page


@pytest.fixture(scope="session")
def server_url() -> Generator[str, None, None]:
    """Start the FastAPI app in a subprocess and yield its base URL.

    Uses subprocess.Popen instead of multiprocessing.Process to avoid
    Windows 'spawn' pickling issues.
    """
    host, port = "127.0.0.1", 18765
    env = {**os.environ, "WHISPER_MODEL": "tiny"}

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "backend.server:app",
            "--host", host,
            "--port", str(port),
            "--log-level", "warning",
        ],
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    )

    # Wait for the server to be reachable
    import httpx

    url = f"http://{host}:{port}"
    deadline = time.time() + 120  # generous timeout for model download + loading
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/api/config", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        proc.kill()
        pytest.fail("Server did not start within 120 seconds")

    yield url

    proc.kill()
    proc.wait(timeout=5)


@pytest.fixture()
def page(server_url: str, page: Page) -> Page:
    """Navigate a Playwright page to the server URL."""
    page.goto(server_url, wait_until="networkidle")
    return page
