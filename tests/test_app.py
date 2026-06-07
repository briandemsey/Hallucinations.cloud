import time
import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:8501"
TEST_EMAIL = "bdemsey@gmail.com"
TEST_PHONE = "+19492911422"
TEST_CODE = "123456"
TEST_QUERY = "What is the capital of France?"


def wait(seconds=2):
    time.sleep(seconds)


def test_full_signin_and_query(page: Page):
    # Load app
    page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
    wait(3)

    # Step 1: Check Start Free Trial
    page.click("text=Start Free Trial")
    wait(2)

    # Step 2: Enter email and save
    email_field = page.locator("input[placeholder='you@example.com']")
    email_field.fill(TEST_EMAIL)
    email_field.press("Enter")
    wait(2)
    page.locator("button:has-text('Save')").click(timeout=10000)
    wait(2)

    # Step 3: Enter phone and send
    phone_field = page.locator("input[placeholder='+1 (555) 123-4567']")
    phone_field.fill(TEST_PHONE)
    phone_field.press("Enter")
    wait(2)
    page.locator("button:has-text('Send')").click(timeout=10000)
    wait(3)

    # Step 4: Enter SMS code and verify
    code_field = page.locator("input[placeholder='123456']")
    code_field.fill(TEST_CODE)
    code_field.press("Enter")
    wait(2)
    page.locator("button:has-text('Verify')").click(timeout=10000)
    wait(3)

    # Step 5: Enter and submit query
    query_area = page.locator("textarea[placeholder='Ask anything...']")
    query_area.fill(TEST_QUERY)
    wait(1)
    page.locator("button:has-text('Submit')").click(timeout=10000)
    wait(2)

    # Wait for results (up to 90 seconds)
    wait(90)

    # Verify question is visible on results page
    content = page.inner_text("body").encode("ascii", errors="replace").decode()
    assert "capital of France" in content or "Your question" in content, (
        f"Question not visible on results page. Preview: {content[:500]}"
    )
    print("SUCCESS: Question visible on results page")
