# Screenshot Generation Guide

This guide explains how to generate consistent, high-quality screenshots for ComfyAI documentation using our Playwright-based testing suite.

## 🎯 Quick Start

### 1. Install Dependencies
```bash
# Install Playwright dependencies
pip install -r requirements-playwright.txt

# Install Playwright browsers (one-time setup)
python scripts/generate_screenshots.py --setup
```

### 2. Start the Server
```bash
# Start ComfyAI server
./deploy.sh run

# Server should be running at http://localhost:8000
```

### 3. Generate Screenshots
```bash
# GitHub README optimized (800x600)
python scripts/generate_screenshots.py readme

# Complete interface tour
python scripts/generate_screenshots.py interface_tour

# All available resolutions
python scripts/generate_screenshots.py all_resolutions

# List all configurations
python scripts/generate_screenshots.py --list
```

## 📐 Screenshot Configurations

### Predefined Resolutions

| Configuration | Resolution | Render Size | Purpose |
|---------------|------------|-------------|---------|
| `github_readme` | 800×600 | 1600×1200 (2x DPI) | Optimal for GitHub README images |
| `docs` | 1200×900 | 2400×1800 (2x DPI) | Documentation pages |
| `mobile` | 375×667 | 750×1334 (2x DPI) | Mobile responsive testing |
| `desktop` | 1920×1080 | 1920×1080 (1x DPI) | Full desktop experience |
| `print` | 800×600 | 3200×2400 (4x DPI) | Ultra high-quality for presentations |

### Available Configurations

| Name | Description | Output |
|------|-------------|--------|
| `readme` | GitHub README optimized | Home, Model Manager, Vision Test |
| `docs` | Documentation pages | High-res interface screenshots |
| `all_resolutions` | Every configured resolution | All interfaces at all sizes |
| `interface_tour` | Complete UI walkthrough | Every major interface |
| `custom` | Custom resolution tests | Parametrized size testing |
| `print_quality` | Ultra high-quality (4x DPI) | Presentation-ready screenshots |

## 🔧 Usage Examples

### Basic Screenshot Generation
```bash
# Generate README screenshots with cleanup
python scripts/generate_screenshots.py readme --clean --verbose

# Generate without server check (useful for CI)
python scripts/generate_screenshots.py readme --no-server-check
```

### Advanced Usage
```bash
# Run specific test classes directly with pytest
python -m pytest tests/test_playwright_screenshots.py::TestMainInterface -v

# Generate screenshots at specific resolution
python -m pytest "tests/test_playwright_screenshots.py::TestCustomScreenshots::test_custom_resolution_home[resolution0]" -v

# Run with visible browser for debugging
python -m pytest tests/test_playwright_screenshots.py --headed -v
```

### Custom Resolution Testing
```bash
# Test multiple custom resolutions
python -m pytest tests/test_playwright_screenshots.py::TestCustomScreenshots -v
```

## 📁 Output Structure

Screenshots are saved to the `screenshots/` directory with descriptive names:

```
screenshots/
├── 01_home_page_800x600.png              # Home page at GitHub README size
├── 02_swagger_docs_800x600.png           # API documentation
├── 03_model_manager_overview_800x600.png  # Model management interface
├── 04_model_manager_expanded_800x600.png  # Expanded model details
├── 05_model_card_detail_800x600.png      # Individual model card
├── 06_vision_test_interface_800x600.png  # Vision testing page
├── 07_vision_test_upload_area_800x600.png # Image upload focus
├── 08_vision_test_configured_800x600.png # Configured test form
├── home_page_github_readme_800x600.png   # Multi-resolution variants
├── home_page_docs_1200x900.png
├── home_page_mobile_375x667.png
└── home_page_desktop_1920x1080.png
```

## 🎨 Customization

### Adding New Screenshot Tests

1. **Add test method to existing class:**
```python
class TestMainInterface:
    def test_new_feature_page(self, screenshot_helper: ScreenshotHelper):
        """Screenshot a new feature page."""
        screenshot_helper.page.goto("http://localhost:8000/new-feature")
        screenshot_helper.wait_for_load()
        screenshot_helper.take_screenshot("new_feature_page")
```

2. **Add new test class:**
```python
class TestNewFeatures:
    """Test new feature interfaces."""
    
    def test_feature_overview(self, screenshot_helper: ScreenshotHelper):
        screenshot_helper.page.goto("http://localhost:8000/features")
        screenshot_helper.wait_for_load()
        screenshot_helper.take_screenshot("features_overview")
```

### Custom Screenshot Configuration

Create a new configuration in `scripts/generate_screenshots.py`:

```python
CONFIGURATIONS["my_config"] = {
    "description": "My custom configuration",
    "tests": [
        "tests/test_playwright_screenshots.py::TestMainInterface::test_home_page",
        "tests/test_playwright_screenshots.py::TestNewFeatures",
    ]
}
```

### Custom Resolutions

Add new resolutions to the config:

```python
SCREENSHOT_CONFIG["my_size"] = {"width": 1440, "height": 900}
```

## 🔍 Testing Strategy

### Test Organization

- **TestMainInterface**: Core application pages (home, swagger)
- **TestModelManagement**: Model management and downloading interfaces
- **TestVisionInterface**: Vision testing and upload interfaces  
- **TestMultiResolution**: Same tests across multiple screen sizes
- **TestCustomScreenshots**: Parametrized resolution testing
- **TestServerRequired**: Tests requiring active server (marked with skip)

### Screenshot Helper Features

```python
# Basic screenshot
helper.take_screenshot("page_name")

# Full page screenshot (scrolls to capture all content)
helper.take_screenshot("page_name", full_page=True)

# Screenshot specific element
helper.take_screenshot("element_name", element_selector=".model-card")

# Custom viewport before screenshot
helper.page.set_viewport_size({"width": 1024, "height": 768})
helper.take_screenshot("custom_size")
```

## 🚀 CI/CD Integration

### GitHub Actions Example

```yaml
name: Generate Screenshots
on:
  push:
    branches: [main, develop]

jobs:
  screenshots:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install system dependencies (Linux)
        run: |
          sudo apt-get update
          sudo apt-get install -y xvfb
      
      - name: Install Python dependencies
        run: |
          pip install -r requirements-playwright-linux.txt
          python scripts/generate_screenshots.py --setup
      
      - name: Start server
        run: |
          ./deploy.sh build
          ./deploy.sh run &
          sleep 10  # Wait for server startup
      
      - name: Generate screenshots
        run: python scripts/generate_screenshots.py readme --no-server-check
      
      - name: Upload screenshots
        uses: actions/upload-artifact@v3
        with:
          name: screenshots
          path: screenshots/
```

### Cross-Platform Usage

The system automatically detects your platform and adjusts accordingly:

**macOS (Development):**
```bash
pip install -r requirements-playwright.txt  # Core dependencies only
python scripts/generate_screenshots.py readme
```

**Linux (CI/CD):**
```bash
sudo apt-get install xvfb  # Virtual display server
pip install -r requirements-playwright-linux.txt  # Includes pytest-xvfb
python scripts/generate_screenshots.py readme  # Auto-uses --xvfb flag
```

**Windows:**
```bash
pip install -r requirements-playwright.txt  # Native headless support
python scripts/generate_screenshots.py readme
```

## 🛠️ Troubleshooting

### Common Issues

**Server not running:**
```bash
# Check if server is accessible
curl http://localhost:8000/

# Start server if needed
./deploy.sh run
```

**Playwright browsers not installed:**
```bash
# Install browsers
python scripts/generate_screenshots.py --setup

# Or manually
playwright install chromium
```

**Permission errors:**
```bash
# Make script executable
chmod +x scripts/generate_screenshots.py
```

**Screenshots directory not created:**
```bash
# Create manually if needed
mkdir screenshots
```

### Debug Mode

Run tests with visible browser for debugging:
```bash
python -m pytest tests/test_playwright_screenshots.py --headed --slowmo=1000 -v
```

### Headless Issues on Linux

Install virtual display dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install xvfb

# Use with pytest-xvfb
python -m pytest tests/test_playwright_screenshots.py --xvfb
```

## 📋 Best Practices

1. **Consistent Naming**: Use descriptive screenshot names with dimensions
2. **Wait for Load**: Always wait for full page load and animations
3. **Clean Runs**: Use `--clean` to remove old screenshots before generating new ones
4. **Multiple Resolutions**: Test key interfaces at multiple screen sizes
5. **Element Focus**: Use element selectors for specific UI component screenshots
6. **Server Health**: Check server is running and responsive before screenshot generation
7. **CI Integration**: Automate screenshot generation in CI/CD pipelines
8. **Version Control**: Consider excluding screenshots from git or using Git LFS for large files

## 🎯 GitHub README Optimization

For optimal GitHub README display:
- Use 800×600 resolution (fits most screens without scrolling)
- Focus on key UI elements and workflows
- Ensure text is readable at small sizes
- Use descriptive filenames for easy maintenance
- Consider dark/light theme variations if applicable

This system ensures consistent, professional screenshots for all ComfyAI documentation! 📸✨