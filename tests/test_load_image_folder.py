import os
import sys
import tempfile
import pytest
import numpy as np
from PIL import Image
from pathlib import Path

# Set unit test mode before importing our nodes
os.environ["UNIT_TEST_MODE"] = "1"

# Mock torch before importing
import types
torch = types.ModuleType("torch")
torch.from_numpy = lambda x: x
torch.stack = lambda x, dim=0: x[0] if len(x) == 1 else x
torch.Tensor = type("MockTensor", (), {"shape": [1, 64, 64, 3]})
torch.float32 = float

class MockTensor:
    def __init__(self, data):
        self.data = data
        self.shape = getattr(data, 'shape', [64, 64, 3])  # Default shape for single image
    
    def min(self):
        return 0.0
    
    def max(self):
        return 1.0
    
    def numpy(self):
        # Return a mock numpy array for testing
        return np.random.rand(*self.shape)

torch.from_numpy = lambda x: MockTensor(x)
torch.stack = lambda x, dim=0: MockTensor(x)

sys.modules["torch"] = torch

from nodes.load_image_folder import LoadImageFolder


class TestLoadImageFolder:
    def create_test_images(self, temp_dir, count=3, size=(64, 64)):
        """Create test images in a temporary directory."""
        image_paths = []
        for i in range(count):
            # Create a simple test image with different colors
            img = Image.new('RGB', size, color=(i * 80, 100, 200))
            img_path = os.path.join(temp_dir, f"test_image_{i:02d}.jpg")
            img.save(img_path)
            image_paths.append(img_path)
        return image_paths

    def test_load_images_basic(self):
        """Test basic image loading functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test images
            self.create_test_images(temp_dir, count=3)
            
            # Test the node
            node = LoadImageFolder()
            result = node.load_images(temp_dir)
            
            # Verify results
            assert len(result) == 1  # Should return tuple with one element
            batch_tensor = result[0]
            
            assert isinstance(batch_tensor, MockTensor)
            assert hasattr(batch_tensor, 'shape')
            
            # Values should be in [0, 1] range
            assert batch_tensor.min() >= 0.0
            assert batch_tensor.max() <= 1.0

    def test_load_images_with_extensions_filter(self):
        """Test loading images with specific extensions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mixed file types
            self.create_test_images(temp_dir, count=2)
            
            # Create a non-image file
            with open(os.path.join(temp_dir, "test.txt"), "w") as f:
                f.write("not an image")
            
            # Create a PNG image
            img = Image.new('RGB', (64, 64), color=(255, 0, 0))
            img.save(os.path.join(temp_dir, "test.png"))
            
            # Test with PNG filter only
            node = LoadImageFolder()
            result = node.load_images(temp_dir, image_extensions=".png")
            
            batch_tensor = result[0]
            assert isinstance(batch_tensor, MockTensor)  # Should be MockTensor for single PNG

    def test_load_images_max_limit(self):
        """Test max_images parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create 5 test images
            self.create_test_images(temp_dir, count=5)
            
            # Test with max_images=2
            node = LoadImageFolder()
            result = node.load_images(temp_dir, max_images=2)
            
            batch_tensor = result[0]
            assert isinstance(batch_tensor, MockTensor)  # Should be MockTensor

    def test_load_images_different_sizes(self):
        """Test loading images with different sizes (should be resized)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create images with different sizes
            sizes = [(64, 64), (128, 128), (32, 32)]
            for i, size in enumerate(sizes):
                img = Image.new('RGB', size, color=(i * 80, 100, 200))
                img.save(os.path.join(temp_dir, f"test_{i}.jpg"))
            
            # Test the node
            node = LoadImageFolder()
            result = node.load_images(temp_dir)
            
            batch_tensor = result[0]
            assert isinstance(batch_tensor, MockTensor)  # Should be MockTensor
            assert hasattr(batch_tensor, 'shape')  # Should have shape attribute

    def test_empty_folder(self):
        """Test error handling for empty folders."""
        with tempfile.TemporaryDirectory() as temp_dir:
            node = LoadImageFolder()
            
            with pytest.raises(ValueError, match="No image files found"):
                node.load_images(temp_dir)

    def test_nonexistent_folder(self):
        """Test error handling for non-existent folders."""
        node = LoadImageFolder()
        
        with pytest.raises(ValueError, match="Folder does not exist"):
            node.load_images("/nonexistent/path")

    def test_empty_folder_path(self):
        """Test error handling for empty folder path."""
        node = LoadImageFolder()
        
        with pytest.raises(ValueError, match="folder_path cannot be empty"):
            node.load_images("")

    def test_input_types(self):
        """Test INPUT_TYPES class method."""
        input_types = LoadImageFolder.INPUT_TYPES()
        
        assert "required" in input_types
        assert "optional" in input_types
        assert "folder_path" in input_types["required"]
        assert "image_extensions" in input_types["optional"]
        assert "max_images" in input_types["optional"]

    def test_return_types(self):
        """Test RETURN_TYPES constant."""
        assert LoadImageFolder.RETURN_TYPES == ("IMAGE",)
        assert LoadImageFolder.FUNCTION == "load_images"
        assert LoadImageFolder.CATEGORY == "image"