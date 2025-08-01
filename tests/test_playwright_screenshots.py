#!/usr/bin/env python3
"""Playwright tests for generating consistent screenshots for documentation."""

import pytest
from playwright.sync_api import sync_playwright, Page, Browser
import os
from pathlib import Path
from typing import Tuple
import time

# Screenshot configuration
SCREENSHOT_CONFIG = {
    # GitHub README optimal resolution - render larger for better DPI
    "github_readme": {
        "width": 1600, "height": 1200,  # 2x resolution for crisp rendering
        "scale": 0.5,  # Scale down to 800x600 for final output
        "device_scale_factor": 2  # High DPI rendering
    },
    # Documentation pages - high DPI
    "docs": {
        "width": 2400, "height": 1800,  # 2x resolution 
        "scale": 0.5,  # Scale down to 1200x900
        "device_scale_factor": 2
    },
    # Mobile view - high DPI
    "mobile": {
        "width": 750, "height": 1334,  # 2x resolution
        "scale": 0.5,  # Scale down to 375x667
        "device_scale_factor": 2
    },
    # Desktop view - standard DPI (already high resolution)
    "desktop": {"width": 1920, "height": 1080},
    # Ultra high quality for print/presentations
    "print": {
        "width": 3200, "height": 2400,  # 4x resolution
        "scale": 0.25,  # Scale down to 800x600
        "device_scale_factor": 3
    },
}

# Default config for most screenshots
DEFAULT_CONFIG = SCREENSHOT_CONFIG["github_readme"]

# Server URLs
BASE_URL = "http://localhost:8000"
URLS = {
    "home": f"{BASE_URL}/",
    "manage": f"{BASE_URL}/manage/ui/",
    "vision_test": f"{BASE_URL}/test",
    "swagger": f"{BASE_URL}/docs",
    "model_manager": f"{BASE_URL}/ui",
}

# Output directory
SCREENSHOTS_DIR = Path(__file__).parent.parent / "screenshots"


class ScreenshotHelper:
    """Helper class for consistent screenshot generation."""
    
    def __init__(self, page: Page, config: dict = None):
        self.page = page
        self.config = config or DEFAULT_CONFIG
        
    def setup_viewport(self):
        """Set up the viewport with configured dimensions."""
        # Handle both simple and high-DPI configs
        if isinstance(self.config, dict) and "width" in self.config:
            viewport_config = {
                "width": self.config["width"],
                "height": self.config["height"]
            }
            # Add device scale factor if specified
            if "device_scale_factor" in self.config:
                viewport_config["device_scale_factor"] = self.config["device_scale_factor"]
            
            self.page.set_viewport_size(viewport_config)
        else:
            # Legacy simple config
            self.page.set_viewport_size(self.config)
        
    def wait_for_load(self, timeout: int = 5000):
        """Wait for page to fully load."""
        self.page.wait_for_load_state("networkidle", timeout=timeout)
        # Additional wait for any animations/transitions
        time.sleep(0.5)
        
    def take_screenshot(self, name: str, full_page: bool = False, 
                       element_selector: str = None) -> Path:
        """Take a screenshot with consistent naming and location."""
        SCREENSHOTS_DIR.mkdir(exist_ok=True)
        
        # Determine output dimensions and scaling
        if isinstance(self.config, dict) and "scale" in self.config:
            # High-DPI config with scaling
            render_width, render_height = self.config["width"], self.config["height"]
            output_width = int(render_width * self.config["scale"])
            output_height = int(render_height * self.config["scale"])
            filename = f"{name}_{output_width}x{output_height}.png"
        else:
            # Simple config
            render_width, render_height = self.config["width"], self.config["height"]
            filename = f"{name}_{render_width}x{render_height}.png"
        
        filepath = SCREENSHOTS_DIR / filename
        
        if element_selector:
            # Screenshot specific element
            element = self.page.locator(element_selector)
            element.screenshot(path=str(filepath))
        else:
            # Screenshot full page or viewport
            screenshot_options = {
                "path": str(filepath),
                "full_page": full_page
            }
            
            # Add scaling if needed for high-DPI output
            if isinstance(self.config, dict) and "scale" in self.config:
                # Import PIL for resizing
                from PIL import Image
                
                # Take screenshot at high resolution
                temp_path = filepath.with_suffix('.temp.png')
                self.page.screenshot(
                    path=str(temp_path),
                    full_page=full_page
                )
                
                # Resize to target dimensions
                with Image.open(temp_path) as img:
                    output_size = (output_width, output_height)
                    resized_img = img.resize(output_size, Image.Resampling.LANCZOS)
                    resized_img.save(filepath, optimize=True, quality=95)
                
                # Clean up temp file
                temp_path.unlink()
            else:
                # Standard screenshot
                self.page.screenshot(**screenshot_options)
            
        print(f"Screenshot saved: {filepath}")
        return filepath


@pytest.fixture(scope="session")
def browser():
    """Create a browser instance for the test session."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(browser: Browser):
    """Create a new page for each test."""
    page = browser.new_page()
    yield page
    page.close()


@pytest.fixture
def screenshot_helper(page: Page):
    """Create a screenshot helper with default config."""
    helper = ScreenshotHelper(page, DEFAULT_CONFIG)
    helper.setup_viewport()
    return helper


@pytest.fixture(params=list(SCREENSHOT_CONFIG.keys()))
def multi_resolution_helper(page: Page, request):
    """Create screenshot helpers for multiple resolutions."""
    config_name = request.param
    config = SCREENSHOT_CONFIG[config_name]
    helper = ScreenshotHelper(page, config)
    helper.setup_viewport()
    helper.config_name = config_name
    return helper


@pytest.mark.screenshot
class TestMainInterface:
    """Test main application interfaces."""
    
    def test_home_page(self, screenshot_helper: ScreenshotHelper):
        """Screenshot the main landing page."""
        screenshot_helper.page.goto(URLS["home"])
        screenshot_helper.wait_for_load()
        screenshot_helper.take_screenshot("01_home_page")
        
    def test_swagger_docs(self, screenshot_helper: ScreenshotHelper):
        """Screenshot the API documentation."""
        screenshot_helper.page.goto(URLS["swagger"])
        screenshot_helper.wait_for_load()
        screenshot_helper.take_screenshot("02_swagger_docs", full_page=True)


@pytest.mark.screenshot
class TestModelManagement:
    """Test model management interfaces."""
    
    def test_unified_model_manager(self, screenshot_helper: ScreenshotHelper):
        """Screenshot the unified model management interface."""
        screenshot_helper.page.goto(URLS["manage"])
        screenshot_helper.wait_for_load()
        
        # Take initial screenshot
        screenshot_helper.take_screenshot("03_model_manager_overview")
        
        # Expand a model section if available
        try:
            chat_models = screenshot_helper.page.locator("h3:has-text('Chat Models')")
            if chat_models.is_visible():
                chat_models.click()
                time.sleep(0.5)
                screenshot_helper.take_screenshot("04_model_manager_expanded")
        except Exception:
            pass  # Skip if no expandable sections
            
    def test_model_manager_details(self, screenshot_helper: ScreenshotHelper):
        """Screenshot model details and download states."""
        screenshot_helper.page.goto(URLS["manage"])
        screenshot_helper.wait_for_load()
        
        # Look for model cards
        try:
            model_cards = screenshot_helper.page.locator(".model-card").first
            if model_cards.is_visible():
                screenshot_helper.take_screenshot(
                    "05_model_card_detail", 
                    element_selector=".model-card"
                )
        except Exception:
            # Fallback to full page if no specific cards
            screenshot_helper.take_screenshot("05_model_manager_fallback")


@pytest.mark.screenshot
class TestVisionInterface:
    """Test vision testing interface."""
    
    def test_vision_test_page(self, screenshot_helper: ScreenshotHelper):
        """Screenshot the vision testing interface."""
        screenshot_helper.page.goto(URLS["vision_test"])
        screenshot_helper.wait_for_load()
        
        # Initial page
        screenshot_helper.take_screenshot("06_vision_test_interface")
        
        # Try to show the image upload area more clearly
        try:
            upload_area = screenshot_helper.page.locator("#imageInput")
            if upload_area.is_visible():
                # Focus on the upload area
                upload_area.scroll_into_view_if_needed()
                time.sleep(0.3)
                screenshot_helper.take_screenshot("07_vision_test_upload_area")
        except Exception:
            pass
            
    def test_vision_test_with_sample(self, screenshot_helper: ScreenshotHelper):
        """Test vision interface with sample data (if possible)."""
        screenshot_helper.page.goto(URLS["vision_test"])
        screenshot_helper.wait_for_load()
        
        # Try to interact with form elements
        try:
            # Fill in a sample prompt
            prompt_input = screenshot_helper.page.locator("#prompt")
            if prompt_input.is_visible():
                prompt_input.fill("What is shown in the provided image?")
                
            # Select a model
            model_select = screenshot_helper.page.locator("#modelName")
            if model_select.is_visible():
                model_select.select_option(index=0)
                
            time.sleep(0.3)
            screenshot_helper.take_screenshot("08_vision_test_configured")
        except Exception:
            # If form interaction fails, just screenshot the page
            screenshot_helper.take_screenshot("08_vision_test_basic")


@pytest.mark.screenshot
class TestMultiResolution:
    """Generate screenshots at multiple resolutions."""
    
    def test_home_page_all_resolutions(self, multi_resolution_helper: ScreenshotHelper):
        """Generate home page screenshots at all configured resolutions."""
        multi_resolution_helper.page.goto(URLS["home"])
        multi_resolution_helper.wait_for_load()
        
        config_name = multi_resolution_helper.config_name
        multi_resolution_helper.take_screenshot(f"home_page_{config_name}")
        
    def test_model_manager_all_resolutions(self, multi_resolution_helper: ScreenshotHelper):
        """Generate model manager screenshots at all configured resolutions."""
        multi_resolution_helper.page.goto(URLS["manage"])
        multi_resolution_helper.wait_for_load()
        
        config_name = multi_resolution_helper.config_name
        multi_resolution_helper.take_screenshot(f"model_manager_{config_name}")


@pytest.mark.screenshot
class TestCustomScreenshots:
    """Custom screenshot configurations."""
    
    @pytest.mark.parametrize("resolution", [(800, 600), (1200, 800), (1920, 1080)])
    def test_custom_resolution_home(self, page: Page, resolution: Tuple[int, int]):
        """Test home page at custom resolution."""
        width, height = resolution
        config = {"width": width, "height": height}
        helper = ScreenshotHelper(page, config)
        helper.setup_viewport()
        
        helper.page.goto(URLS["home"])
        helper.wait_for_load()
        helper.take_screenshot(f"home_custom_{width}x{height}")


@pytest.mark.screenshot
@pytest.mark.skip(reason="Only run when server is available")
class TestServerRequired:
    """Tests that require the server to be running."""
    
    def test_api_response_screenshots(self, screenshot_helper: ScreenshotHelper):
        """Screenshot API responses in browser."""
        # Test API endpoint directly in browser
        api_url = f"{BASE_URL}/v1/models"
        screenshot_helper.page.goto(api_url)
        screenshot_helper.wait_for_load()
        screenshot_helper.take_screenshot("09_api_models_response")


def test_screenshot_directory_creation():
    """Ensure screenshot directory is created."""
    SCREENSHOTS_DIR.mkdir(exist_ok=True)
    assert SCREENSHOTS_DIR.exists()
    assert SCREENSHOTS_DIR.is_dir()


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_playwright_screenshots.py -v
    # Or just specific tests: python -m pytest tests/test_playwright_screenshots.py::TestMainInterface -v
    # Generate all resolutions: python -m pytest tests/test_playwright_screenshots.py::TestMultiResolution -v
    pytest.main([__file__, "-v"])