"""Regenerate the app screenshots in docs/img (1400 px wide) from a running app; also a browser smoke test.
Usage: streamlit run demo/streamlit_app.py --server.headless true &   then   python scripts/screenshots.py [URL]
Needs playwright (pip install playwright && playwright install chromium)."""
import pathlib
import sys
import tempfile

from playwright.sync_api import expect, sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from c2c import demo  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = ROOT / "docs/img"


def shot(page, name):
    """Grow the viewport to the content height so the whole tab fits, then save."""
    page.wait_for_timeout(800)
    h = page.evaluate("Math.max(...[...document.querySelectorAll('[data-testid=stMain], "
                      "[data-testid=stMainBlockContainer]')].map(e => e.scrollHeight))")
    page.set_viewport_size({"width": 1400, "height": max(900, h + 40)})
    page.wait_for_timeout(800)
    page.screenshot(path=str(OUT / name))
    page.set_viewport_size({"width": 1400, "height": 1000})
    print("saved", OUT / name)


def main():
    OUT.mkdir(exist_ok=True)
    sample = pathlib.Path(tempfile.mkdtemp()) / "corr2cause_sample.csv"
    sample.write_bytes(demo.sample_csv())
    with sync_playwright() as p:
        page = p.chromium.launch().new_page(viewport={"width": 1400, "height": 1000})
        page.goto(URL)
        expect(page.get_by_role("heading", name="Corr2Cause: neuro-symbolic causal inference")).to_be_visible(
            timeout=60_000)

        page.get_by_test_id("stSelectbox").first.click()
        page.get_by_role("option").first.click()
        expect(page.get_by_label("Hypothesis")).to_have_value(demo.EXAMPLES[0][1])
        page.get_by_role("button", name="Check").click()
        expect(page.get_by_text("Parsed triple")).to_be_visible(timeout=180_000)
        shot(page, "app_corr2cause.png")

        page.get_by_role("tab", name="Batch").click()
        page.locator("input[type=file]").set_input_files(str(sample))
        expect(page.get_by_text("3 rows scored")).to_be_visible(timeout=180_000)
        with page.expect_download() as download:
            page.get_by_role("button", name="Download results CSV").click()
        header = pathlib.Path(download.value.path()).read_text().splitlines()[0]
        assert header.startswith("premise,hypothesis,label,aml_p"), header
        shot(page, "app_batch.png")

        page.get_by_role("tab", name="Free text").click()
        expect(page.get_by_text("ATE").first).to_be_visible()
        shot(page, "app_freetext.png")


if __name__ == "__main__":
    main()
