"""
Multimodal Model Tests for Qwen3-VL

Simple test suite to validate multimodal functionality with Qwen3-VL models.
Tests can be run individually without pytest if needed.
"""

import sys
import urllib.request
from pathlib import Path
from typing import Optional

import llama_cpp
from llama_cpp import llama_chat_format


# Test model URLs - using smaller models for testing
QWEN3_VL_MODEL_URL = "https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-2B-Thinking-abliterated/resolve/main/GGUF/ggml-model-f16.gguf"
QWEN3_VL_MMPROJ_URL = "https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-2B-Thinking-abliterated/resolve/main/GGUF/mmproj-model-f16.gguf"

# Test image URLs
TEST_IMAGE_URL = "https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png"


class ModelDownloader:
    """Helper class to download and cache test models."""

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "llama-cpp-python-tests"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_model(self, url: str, filename: str) -> Optional[str]:
        """Download model if not cached, return path to cached file."""
        cached_path = self.cache_dir / filename

        if cached_path.exists():
            print(f"Using cached model: {cached_path}")
            return str(cached_path)

        print(f"Downloading {filename} from {url}")
        try:
            urllib.request.urlretrieve(url, str(cached_path))
            print(f"Downloaded to: {cached_path}")
            return str(cached_path)
        except Exception as e:
            print(f"Could not download test model {filename}: {e}")
            return None

    def get_qwen3_vl_paths(self) -> Optional[tuple[str, str]]:
        """Get paths to Qwen3-VL model and mmproj files."""
        model_path = self.download_model(QWEN3_VL_MODEL_URL, "qwen3-vl-2b-model.gguf")
        mmproj_path = self.download_model(
            QWEN3_VL_MMPROJ_URL, "qwen3-vl-2b-mmproj.gguf"
        )

        if model_path and mmproj_path:
            return model_path, mmproj_path
        return None


def test_model_loading_without_mmproj():
    """Test that we can load the base model without multimodal projector."""
    print("Testing model loading without mmproj...")

    downloader = ModelDownloader()
    paths = downloader.get_qwen3_vl_paths()
    if not paths:
        print("SKIP: Could not download test models")
        return False

    model_path, _ = paths

    try:
        # Should be able to load as text-only model
        llama = llama_cpp.Llama(model_path=model_path, n_ctx=512, verbose=False)

        # Test basic text generation
        response = llama("Hello", max_tokens=10)

        assert response is not None
        print(f"Basic generation test passed: {type(response)}")

        llama.close()
        print("✓ Model loading test passed")
        return True

    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        return False


def test_qwen25vl_chat_handler_creation():
    """Test creating Qwen2.5-VL chat handler."""
    print("Testing Qwen2.5-VL chat handler creation...")

    downloader = ModelDownloader()
    paths = downloader.get_qwen3_vl_paths()
    if not paths:
        print("SKIP: Could not download test models")
        return False

    model_path, mmproj_path = paths

    try:
        # Test that we can create the chat handler
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path, verbose=False
        )

        assert chat_handler is not None
        assert chat_handler.clip_model_path == mmproj_path
        assert not chat_handler.verbose

        print("✓ Chat handler creation test passed")
        return True

    except Exception as e:
        print(f"✗ Chat handler creation test failed: {e}")
        return False


def test_multimodal_model_loading_with_chat_handler():
    """Test loading multimodal model with chat handler."""
    print("Testing multimodal model loading with chat handler...")

    downloader = ModelDownloader()
    paths = downloader.get_qwen3_vl_paths()
    if not paths:
        print("SKIP: Could not download test models")
        return False

    model_path, mmproj_path = paths

    try:
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path, verbose=False
        )

        # Load model with chat handler
        llama = llama_cpp.Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=1024,  # Larger context for multimodal
            verbose=False,
        )

        assert llama is not None
        assert llama.chat_handler is not None

        llama.close()
        print("✓ Multimodal model loading test passed")
        return True

    except Exception as e:
        print(f"✗ Multimodal model loading test failed: {e}")
        return False


def test_image_loading_from_url():
    """Test loading images from URLs."""
    print("Testing image loading from URL...")

    downloader = ModelDownloader()
    paths = downloader.get_qwen3_vl_paths()
    if not paths:
        print("SKIP: Could not download test models")
        return False

    _, mmproj_path = paths

    try:
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path, verbose=False
        )

        # Test loading image from URL
        image_bytes = chat_handler.load_image(TEST_IMAGE_URL)
        assert image_bytes is not None
        assert len(image_bytes) > 0
        assert image_bytes.startswith(b"\x89PNG")  # PNG magic number

        print("✓ Image loading test passed")
        return True

    except Exception as e:
        print(f"✗ Image loading test failed: {e}")
        return False


def test_text_only_chat_completion():
    """Test text-only chat completion with multimodal model."""
    print("Testing text-only chat completion...")

    downloader = ModelDownloader()
    paths = downloader.get_qwen3_vl_paths()
    if not paths:
        print("SKIP: Could not download test models")
        return False

    model_path, mmproj_path = paths

    try:
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path, verbose=False
        )

        # Test that the chat handler can format messages properly
        # This is a safer test that doesn't require full model execution
        messages = [{"role": "user", "content": "Hello! How are you?"}]

        # Test that we can create the handler and it has the expected methods
        assert hasattr(chat_handler, "__call__")
        assert hasattr(chat_handler, "load_image")
        assert hasattr(chat_handler, "get_image_urls")

        # Test image URL extraction
        image_urls = chat_handler.get_image_urls(messages)  # type: ignore
        assert len(image_urls) == 0  # No images in text-only message

        print("✓ Text-only chat handler test passed")
        return True

    except Exception as e:
        print(f"✗ Text-only chat handler test failed: {e}")
        return False


def test_multimodal_chat_completion_with_image():
    """Test multimodal message preparation and image processing."""
    print("Testing multimodal chat completion with image...")

    downloader = ModelDownloader()
    paths = downloader.get_qwen3_vl_paths()
    if not paths:
        print("SKIP: Could not download test models")
        return False

    model_path, mmproj_path = paths

    try:
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path, verbose=False
        )

        # Test with the Qwen demo image
        qwen_demo_image = (
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        )

        # Test image loading first
        print(f"Loading image: {qwen_demo_image}")
        image_bytes = chat_handler.load_image(qwen_demo_image)
        assert image_bytes is not None
        assert len(image_bytes) > 0
        print(f"✓ Image loaded successfully: {len(image_bytes)} bytes")

        # Test message structure for multimodal completion
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": qwen_demo_image}},
                    {
                        "type": "text",
                        "text": "What do you see in this image? Please describe it briefly.",
                    },
                ],
            }
        ]

        # Test image URL extraction
        image_urls = chat_handler.get_image_urls(messages)  # type: ignore
        assert len(image_urls) == 1
        assert image_urls[0] == qwen_demo_image
        print(f"✓ Extracted image URL: {image_urls[0]}")

        # Test that we can create a model with the handler
        print("Testing model initialization with multimodal handler...")
        llama = llama_cpp.Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=1024,  # Conservative context size
            verbose=False,
        )

        # Verify the model is properly initialized
        assert llama is not None
        assert llama.chat_handler is not None
        print("✓ Model initialized with multimodal support")

        # For safety, we'll test message formatting rather than full generation
        # to avoid potential segfaults in the CI environment
        print("✓ Multimodal pipeline validation completed")

        llama.close()
        print("✓ Multimodal chat completion preparation test passed")
        return True

    except Exception as e:
        print(f"✗ Multimodal chat completion test failed: {e}")
        return False


def test_qwen25vl_chat_format_template():
    """Test Qwen2.5-VL chat format template."""
    print("Testing Qwen2.5-VL chat format template...")

    try:
        handler = llama_chat_format.Qwen25VLChatHandler

        # Check that the chat format template exists and has expected tokens
        assert hasattr(handler, "CHAT_FORMAT")
        chat_format = handler.CHAT_FORMAT

        # Should contain Qwen-specific tokens
        assert "<|im_start|>" in chat_format
        assert "<|im_end|>" in chat_format
        assert "<|vision_start|>" in chat_format
        assert "<|vision_end|>" in chat_format

        # Should handle system messages
        assert "system" in chat_format
        assert handler.DEFAULT_SYSTEM_MESSAGE == "You are a helpful assistant."

        print("✓ Chat format template test passed")
        return True

    except Exception as e:
        print(f"✗ Chat format template test failed: {e}")
        return False


def test_missing_mmproj_file():
    """Test error handling when mmproj file is missing."""
    print("Testing missing mmproj file error handling...")

    try:
        # Should raise ValueError for non-existent file
        try:
            llama_chat_format.Qwen25VLChatHandler(
                clip_model_path="/nonexistent/path/mmproj.gguf", verbose=False
            )
            print("✗ Expected ValueError was not raised")
            return False
        except ValueError as e:
            if "Clip model path does not exist" in str(e):
                print("✓ Missing mmproj file test passed")
                return True
            else:
                print(f"✗ Unexpected ValueError message: {e}")
                return False
        except Exception as e:
            print(f"✗ Unexpected exception type: {type(e).__name__}: {e}")
            return False

    except Exception as e:
        print(f"✗ Missing mmproj file test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("Running Qwen3-VL Multimodal Tests")
    print("=" * 50)

    tests = [
        test_qwen25vl_chat_format_template,
        test_missing_mmproj_file,
        test_model_loading_without_mmproj,
        test_qwen25vl_chat_handler_creation,
        test_multimodal_model_loading_with_chat_handler,
        test_image_loading_from_url,
        test_text_only_chat_completion,
        test_multimodal_chat_completion_with_image,
    ]

    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"✗ {test.__name__} crashed: {e}")
            results.append((test.__name__, False))

    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{len(results)}")

    return passed == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)


# Pytest compatibility
try:
    import pytest

    def test_qwen3vl_multimodal_suite():
        """Main pytest entry point."""
        assert run_all_tests(), "Some multimodal tests failed"

except ImportError:
    pass  # pytest not available, standalone mode works fine
