"""
Playwright smoke test for LLMVis Dash app.
Usage: python tests/playwright_smoke.py [--url http://127.0.0.1:7860] [--screenshot /tmp/shot.png]
"""
import argparse
import sys
from playwright.sync_api import sync_playwright

def run(url: str, screenshot_path: str | None):
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        print(f"[smoke] Navigating to {url}")
        response = page.goto(url, wait_until="domcontentloaded", timeout=30000)

        assert response and response.status == 200, f"Expected 200, got {response.status if response else 'None'}"
        print(f"[smoke] HTTP {response.status} OK")

        title = page.title()
        assert "Transformer" in title, f"Unexpected title: {title}"
        print(f"[smoke] Title: {title}")

        # Check key UI elements are present
        page.wait_for_selector("#model-dropdown", timeout=10000)
        print("[smoke] Model dropdown found")

        page.wait_for_selector("#prompt-input", timeout=5000)
        print("[smoke] Prompt input found")

        page.wait_for_selector("text=Analyze", timeout=5000)
        print("[smoke] Analyze button found")

        if screenshot_path:
            page.screenshot(path=screenshot_path, full_page=False)
            print(f"[smoke] Screenshot saved to {screenshot_path}")

        browser.close()
        print("[smoke] ✅ All checks passed")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--screenshot", default=None)
    args = parser.parse_args()

    ok = run(args.url, args.screenshot)
    sys.exit(0 if ok else 1)
