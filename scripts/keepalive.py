"""Open the live app in headless Chromium so Streamlit Community Cloud does not put it to sleep.
Clicks the wake-up button if the app is asleep, then waits for the app's title. Exit code 1 on failure.
Usage: APP_URL=https://corr2cause-dl-aml.streamlit.app python scripts/keepalive.py"""
import os
import re
import sys
import time

from playwright.sync_api import Error, sync_playwright

URL = os.environ.get("APP_URL", "https://corr2cause-dl-aml.streamlit.app")
WAKE = "Yes, get this app back up!"
TIMEOUT = 300  # seconds; a cold start reinstalls packages and downloads the model


def app_ready(page) -> bool:
    """The app renders inside an iframe on *.streamlit.app, so check every frame for the title heading."""
    for frame in page.frames:
        try:
            if frame.get_by_role("heading", name=re.compile("Corr2Cause")).count():
                return True
        except Error:  # frame detached while reloading
            pass
    return False


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
        deadline, woke = time.monotonic() + TIMEOUT, False
        while time.monotonic() < deadline:
            if app_ready(page):
                print(f"ok: app is up{' (woken from sleep)' if woke else ''}: {URL}")
                return 0
            wake = page.get_by_role("button", name=WAKE)
            if not woke and wake.count():
                wake.first.click()
                woke = True
                print("app was asleep; clicked the wake-up button")
            page.wait_for_timeout(5_000)
        page.screenshot(path="keepalive_failure.png")
        print(f"error: app title not visible after {TIMEOUT}s: {URL}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
