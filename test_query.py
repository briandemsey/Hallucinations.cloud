"""
Playwright test: full CTC sign-in and query submission.
"""
from playwright.sync_api import sync_playwright
import time

TEST_EMAIL = "bdemsey@gmail.com"
TEST_PHONE = "+19492911422"
TEST_CODE = "123456"
TEST_QUERY = "What is the capital of France?"

def wait_and_screenshot(page, path, wait=2):
    time.sleep(wait)
    page.screenshot(path=path)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})

    print("Opening localhost:8501...")
    page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
    wait_and_screenshot(page, "test_01_loaded.png", 3)
    print("Step 1: Page loaded")

    # Click Start Free Trial checkbox
    page.click("text=Start Free Trial")
    wait_and_screenshot(page, "test_02_checkbox.png", 2)
    print("Step 2: Checked Start Free Trial")

    # Enter email and press Enter to register value in Streamlit
    email_field = page.locator("input[placeholder='you@example.com']")
    email_field.fill(TEST_EMAIL)
    email_field.press("Enter")
    wait_and_screenshot(page, "test_03_email_entered.png", 2)
    print("Step 3: Email entered")

    # Now Save → button should be visible
    page.locator("button:has-text('Save')").click(timeout=10000)
    wait_and_screenshot(page, "test_04_email_saved.png", 2)
    print("Step 4: Email saved")

    # Enter phone and press Enter
    phone_field = page.locator("input[placeholder='+1 (555) 123-4567']")
    phone_field.fill(TEST_PHONE)
    phone_field.press("Enter")
    wait_and_screenshot(page, "test_05_phone_entered.png", 2)
    print("Step 5: Phone entered")

    # Click Send →
    page.locator("button:has-text('Send')").click(timeout=10000)
    wait_and_screenshot(page, "test_06_code_sent.png", 3)
    print("Step 6: Send clicked")

    # Enter SMS code and press Enter
    code_field = page.locator("input[placeholder='123456']")
    code_field.fill(TEST_CODE)
    code_field.press("Enter")
    wait_and_screenshot(page, "test_07_code_entered.png", 2)
    print("Step 7: Code entered")

    # Click Verify →
    page.locator("button:has-text('Verify')").click(timeout=10000)
    wait_and_screenshot(page, "test_08_verified.png", 3)
    print("Step 8: Verified")

    # Enter query in CTC textarea
    query_area = page.locator("textarea[placeholder='Ask anything...']")
    query_area.fill(TEST_QUERY)
    wait_and_screenshot(page, "test_09_query_typed.png", 1)
    print("Step 9: Query typed")

    # Click Submit
    page.locator("button:has-text('Submit')").click(timeout=10000)
    print("Step 10: Query submitted")

    # Screenshot 2 seconds after submit (during spinners)
    wait_and_screenshot(page, "test_10_during_analysis.png", 2)
    print("Step 11: Screenshot during analysis (spinners)")

    # Wait for results
    wait_and_screenshot(page, "test_11_results.png", 60)
    print("Step 12: Screenshot after analysis")

    # Check if question is visible
    content = page.inner_text("body").encode('ascii', errors='replace').decode()
    if "capital of France" in content or "Your question" in content:
        print("SUCCESS: Question is visible on results page")
    else:
        print("FAIL: Question not found on results page")
        print("Page preview:", content[:500])

    browser.close()
    print("Done. Check test_*.png files.")
