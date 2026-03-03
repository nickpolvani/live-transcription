#!/usr/bin/env python3
"""Entry point — starts the uvicorn server and opens the browser."""

import logging
import threading
import time
import webbrowser

import uvicorn

from backend.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


def _open_browser(url: str) -> None:
    """Wait briefly for the server to start, then open the default browser."""
    time.sleep(1.5)
    webbrowser.open(url)


def main() -> None:
    url = f"http://{config.host}:{config.port}"
    print(f"\n  Live Transcription → {url}\n")

    threading.Thread(target=_open_browser, args=(url,), daemon=True).start()

    uvicorn.run(
        "backend.server:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
