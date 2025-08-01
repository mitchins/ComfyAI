#!/usr/bin/env python3
"""Script to generate screenshots for documentation with configurable options."""

import argparse
import subprocess
import sys
from pathlib import Path
import json
import os

# Available screenshot configurations
CONFIGURATIONS = {
    "readme": {
        "description": "GitHub README optimized (800x600)",
        "tests": [
            "tests/test_playwright_screenshots.py::TestMainInterface::test_home_page",
            "tests/test_playwright_screenshots.py::TestModelManagement::test_unified_model_manager",
            "tests/test_playwright_screenshots.py::TestVisionInterface::test_vision_test_page",
        ]
    },
    "docs": {
        "description": "Documentation pages (1200x900)",
        "tests": [
            "tests/test_playwright_screenshots.py::TestMultiResolution::test_home_page_all_resolutions[docs]",
            "tests/test_playwright_screenshots.py::TestMultiResolution::test_model_manager_all_resolutions[docs]",
        ]
    },
    "all_resolutions": {
        "description": "All configured resolutions",
        "tests": [
            "tests/test_playwright_screenshots.py::TestMultiResolution",
        ]
    },
    "interface_tour": {
        "description": "Complete interface tour",
        "tests": [
            "tests/test_playwright_screenshots.py::TestMainInterface",
            "tests/test_playwright_screenshots.py::TestModelManagement",
            "tests/test_playwright_screenshots.py::TestVisionInterface",
        ]
    },
    "custom": {
        "description": "Custom resolution tests",
        "tests": [
            "tests/test_playwright_screenshots.py::TestCustomScreenshots",
        ]
    }
}

def check_server_running(base_url: str = "http://localhost:8000") -> bool:
    """Check if the ComfyAI server is running."""
    try:
        import requests
        response = requests.get(f"{base_url}/", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def setup_playwright():
    """Install Playwright browsers if needed."""
    try:
        result = subprocess.run(
            ["playwright", "install", "chromium"], 
            capture_output=True, 
            text=True,
            check=True
        )
        print("✅ Playwright browsers installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Playwright browsers: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Playwright not found. Install with: pip install -r requirements-playwright.txt")
        return False

def run_screenshot_tests(config_name: str, verbose: bool = False, server_check: bool = True):
    """Run screenshot tests for the specified configuration."""
    
    if config_name not in CONFIGURATIONS:
        print(f"❌ Unknown configuration: {config_name}")
        print(f"Available configurations: {', '.join(CONFIGURATIONS.keys())}")
        return False
        
    config = CONFIGURATIONS[config_name]
    
    # Check server if required
    if server_check and not check_server_running():
        print("❌ ComfyAI server not running at http://localhost:8000")
        print("Start the server with: ./deploy.sh run")
        print("Or skip server check with: --no-server-check")
        return False
    
    print(f"🚀 Running {config['description']} screenshot generation...")
    print(f"📁 Screenshots will be saved to: screenshots/")
    
    # Prepare pytest command
    pytest_args = ["python", "-m", "pytest"]
    pytest_args.extend(config["tests"])
    pytest_args.extend(["-v"] if verbose else [])
    pytest_args.extend(["--tb=short"])  # Shorter tracebacks
    
    try:
        result = subprocess.run(pytest_args, check=True)
        print("✅ Screenshots generated successfully!")
        
        # List generated screenshots
        screenshots_dir = Path("screenshots")
        if screenshots_dir.exists():
            screenshots = list(screenshots_dir.glob("*.png"))
            if screenshots:
                print(f"\n📸 Generated {len(screenshots)} screenshots:")
                for screenshot in sorted(screenshots):
                    print(f"  - {screenshot.name}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Screenshot generation failed with exit code: {e.returncode}")
        return False

def clean_screenshots():
    """Clean up existing screenshots."""
    screenshots_dir = Path("screenshots")
    if screenshots_dir.exists():
        screenshots = list(screenshots_dir.glob("*.png"))
        for screenshot in screenshots:
            screenshot.unlink()
        print(f"🧹 Cleaned up {len(screenshots)} existing screenshots")
    else:
        print("🧹 No existing screenshots to clean")

def list_configurations():
    """List all available screenshot configurations."""
    print("📋 Available screenshot configurations:\n")
    for name, config in CONFIGURATIONS.items():
        print(f"  {name}")
        print(f"    Description: {config['description']}")
        print(f"    Tests: {len(config['tests'])} test(s)")
        print()

def main():
    parser = argparse.ArgumentParser(
        description="Generate screenshots for ComfyAI documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s readme                    # Generate GitHub README screenshots
  %(prog)s docs --verbose           # Generate documentation screenshots with verbose output
  %(prog)s all_resolutions          # Generate screenshots at all resolutions
  %(prog)s interface_tour --clean   # Clean old screenshots and generate interface tour
  %(prog)s --list                   # List all available configurations
  %(prog)s --setup                  # Setup Playwright browsers
        """
    )
    
    parser.add_argument(
        "config", 
        nargs="?",
        choices=list(CONFIGURATIONS.keys()),
        help="Screenshot configuration to run"
    )
    
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all available configurations"
    )
    
    parser.add_argument(
        "--setup", 
        action="store_true",
        help="Setup Playwright browsers"
    )
    
    parser.add_argument(
        "--clean", 
        action="store_true",
        help="Clean existing screenshots before generating new ones"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output from pytest"
    )
    
    parser.add_argument(
        "--no-server-check",
        action="store_true",
        help="Skip checking if the server is running"
    )
    
    args = parser.parse_args()
    
    # Handle special actions
    if args.list:
        list_configurations()
        return
        
    if args.setup:
        success = setup_playwright()
        sys.exit(0 if success else 1)
    
    # Require a configuration if not listing or setting up
    if not args.config:
        parser.print_help()
        print("\n❌ Please specify a configuration or use --list to see options")
        sys.exit(1)
    
    # Clean screenshots if requested
    if args.clean:
        clean_screenshots()
    
    # Run screenshot generation
    success = run_screenshot_tests(
        args.config, 
        verbose=args.verbose,
        server_check=not args.no_server_check
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()