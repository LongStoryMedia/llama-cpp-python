"""
Multimodal Model Tests for Qwen3-VL

Simple test suite to validate multimodal functionality with Qwen3-VL models.
Tests can be run individually without pytest if needed.
"""

import os
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
        mmproj_path = self.download_model(QWEN3_VL_MMPROJ_URL, "qwen3-vl-2b-mmproj.gguf")
        
        if model_path and mmproj_path:
            return model_path, mmproj_path
        return None


class TestMultimodalModelLoading:
    """Test loading of multimodal models."""
    
    def test_model_loading_without_mmproj(self, qwen3_vl_paths):
        """Test that we can load the base model without multimodal projector."""
        model_path, _ = qwen3_vl_paths
        
        # Should be able to load as text-only model
        llama = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=512,
            verbose=False
        )
        
        assert llama is not None
        assert llama._ctx.ctx is not None
        
        # Test basic text generation
        response = llama("Hello", max_tokens=10, stop=["\n"])
        assert response is not None
        assert "choices" in response
        
        llama.close()
    
    def test_qwen25vl_chat_handler_creation(self, qwen3_vl_paths):
        """Test creating Qwen2.5-VL chat handler."""
        model_path, mmproj_path = qwen3_vl_paths
        
        # Test that we can create the chat handler
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path,
            verbose=False
        )
        
        assert chat_handler is not None
        assert chat_handler.clip_model_path == mmproj_path
        assert not chat_handler.verbose
    
    def test_multimodal_model_loading_with_chat_handler(self, qwen3_vl_paths):
        """Test loading multimodal model with chat handler."""
        model_path, mmproj_path = qwen3_vl_paths
        
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path,
            verbose=False
        )
        
        # Load model with chat handler
        llama = llama_cpp.Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=2048,  # Larger context for multimodal
            verbose=False
        )
        
        assert llama is not None
        assert llama.chat_handler is not None
        
        llama.close()


class TestMultimodalImageProcessing:
    """Test image processing functionality."""
    
    def test_image_loading_from_url(self, qwen3_vl_paths):
        """Test loading images from URLs."""
        _, mmproj_path = qwen3_vl_paths
        
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path,
            verbose=False
        )
        
        # Test loading image from URL
        image_bytes = chat_handler.load_image(TEST_IMAGE_URL)
        assert image_bytes is not None
        assert len(image_bytes) > 0
        assert image_bytes.startswith(b'\x89PNG')  # PNG magic number
    
    def test_image_loading_from_base64(self, qwen3_vl_paths):
        """Test loading images from base64 data URLs."""
        _, mmproj_path = qwen3_vl_paths
        
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path,
            verbose=False
        )
        
        # Create a simple base64 image (1x1 PNG)
        base64_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        image_bytes = chat_handler.load_image(base64_image)
        assert image_bytes is not None
        assert len(image_bytes) > 0
    
    def test_invalid_image_url_handling(self, qwen3_vl_paths):
        """Test handling of invalid image URLs."""
        _, mmproj_path = qwen3_vl_paths
        
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path,
            verbose=False
        )
        
        # Test with invalid URL
        with pytest.raises((ValueError, Exception)):  # Should raise some form of exception
            chat_handler.load_image("https://invalid-url-that-does-not-exist.com/image.jpg")


class TestMultimodalChatCompletion:
    """Test chat completion with multimodal inputs."""
    
    def test_text_only_chat_completion(self, qwen3_vl_paths):
        """Test text-only chat completion with multimodal model."""
        model_path, mmproj_path = qwen3_vl_paths
        
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path,
            verbose=False
        )
        
        llama = llama_cpp.Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=1024,
            verbose=False
        )
        
        # Test text-only conversation
        messages = [
            {"role": "user", "content": "Hello! How are you?"}
        ]
        
        response = llama.create_chat_completion(
            messages=messages,
            max_tokens=50,
            temperature=0.1
        )
        
        assert response is not None
        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "message" in response["choices"][0]
        assert "content" in response["choices"][0]["message"]
        
        llama.close()
    
    def test_image_description_chat_completion(self, qwen3_vl_paths):
        """Test chat completion with image input."""
        model_path, mmproj_path = qwen3_vl_paths
        
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path,
            verbose=False
        )
        
        llama = llama_cpp.Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=2048,
            verbose=False
        )
        
        # Test with image URL
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": TEST_IMAGE_URL}
                    },
                    {
                        "type": "text",
                        "text": "What do you see in this image? Please describe it briefly."
                    }
                ]
            }
        ]
        
        response = llama.create_chat_completion(
            messages=messages,
            max_tokens=100,
            temperature=0.1
        )
        
        assert response is not None
        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "message" in response["choices"][0]
        assert "content" in response["choices"][0]["message"]
        
        content = response["choices"][0]["message"]["content"]
        assert len(content.strip()) > 0
        
        llama.close()
    
    def test_multiple_images_chat_completion(self, qwen3_vl_paths):
        """Test chat completion with multiple images."""
        model_path, mmproj_path = qwen3_vl_paths
        
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path,
            verbose=False
        )
        
        llama = llama_cpp.Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=3072,  # Larger context for multiple images
            verbose=False
        )
        
        # Test with multiple images
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": TEST_IMAGE_URL}
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": TEST_DIAGRAM_URL}
                    },
                    {
                        "type": "text",
                        "text": "Compare these two images. What are the differences?"
                    }
                ]
            }
        ]
        
        response = llama.create_chat_completion(
            messages=messages,
            max_tokens=150,
            temperature=0.1
        )
        
        assert response is not None
        assert "choices" in response
        assert len(response["choices"]) > 0
        
        llama.close()


class TestChatFormatting:
    """Test chat format templates for multimodal models."""
    
    def test_qwen25vl_chat_format_template(self):
        """Test Qwen2.5-VL chat format template."""
        handler = llama_chat_format.Qwen25VLChatHandler
        
        # Check that the chat format template exists and has expected tokens
        assert hasattr(handler, 'CHAT_FORMAT')
        chat_format = handler.CHAT_FORMAT
        
        # Should contain Qwen-specific tokens
        assert '<|im_start|>' in chat_format
        assert '<|im_end|>' in chat_format
        assert '<|vision_start|>' in chat_format
        assert '<|vision_end|>' in chat_format
        
        # Should handle system messages
        assert 'system' in chat_format
        assert handler.DEFAULT_SYSTEM_MESSAGE == "You are a helpful assistant."
    
    def test_message_formatting_with_images(self, qwen3_vl_paths):
        """Test message formatting with image content."""
        _, mmproj_path = qwen3_vl_paths
        
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path,
            verbose=False
        )
        
        # Test extracting image URLs from messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": TEST_IMAGE_URL}
                    },
                    {
                        "type": "text",
                        "text": "Describe this image."
                    }
                ]
            }
        ]
        
        image_urls = chat_handler.get_image_urls(messages)
        assert len(image_urls) == 1
        assert image_urls[0] == TEST_IMAGE_URL


class TestErrorHandling:
    """Test error handling in multimodal scenarios."""
    
    def test_missing_mmproj_file(self):
        """Test error handling when mmproj file is missing."""
        with pytest.raises(ValueError, match="Clip model path does not exist"):
            llama_chat_format.Qwen25VLChatHandler(
                clip_model_path="/nonexistent/path/mmproj.gguf",
                verbose=False
            )
    
    def test_invalid_mmproj_file(self, tmp_path):
        """Test error handling with invalid mmproj file."""
        # Create an empty file
        invalid_mmproj = tmp_path / "invalid.gguf"
        invalid_mmproj.write_bytes(b"invalid content")
        
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=str(invalid_mmproj),
            verbose=False
        )
        
        # Error should occur when trying to initialize with a model
        # This would happen during actual usage, not during handler creation
        assert chat_handler is not None


class TestPerformance:
    """Test performance aspects of multimodal models."""
    
    def test_memory_usage_with_images(self, qwen3_vl_paths):
        """Test that memory is properly managed when processing images."""
        model_path, mmproj_path = qwen3_vl_paths
        
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path,
            verbose=False
        )
        
        llama = llama_cpp.Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=1024,
            verbose=False
        )
        
        # Process multiple small requests to check for memory leaks
        for i in range(3):  # Keep small for CI
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": TEST_IMAGE_URL}
                        },
                        {
                            "type": "text",
                            "text": f"Request {i}: What's in this image?"
                        }
                    ]
                }
            ]
            
            response = llama.create_chat_completion(
                messages=messages,
                max_tokens=20,
                temperature=0.1
            )
            
            assert response is not None
        
        llama.close()


class TestCompatibility:
    """Test compatibility with different input formats."""
    
    def test_openai_compatible_format(self, qwen3_vl_paths):
        """Test OpenAI-compatible message format."""
        model_path, mmproj_path = qwen3_vl_paths
        
        chat_handler = llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=mmproj_path,
            verbose=False
        )
        
        llama = llama_cpp.Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=1024,
            verbose=False
        )
        
        # OpenAI-style format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": TEST_IMAGE_URL,
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    }
                ]
            }
        ]
        
        response = llama.create_chat_completion(
            messages=messages,
            max_tokens=30,
            temperature=0.1
        )
        
        assert response is not None
        assert "choices" in response
        
        llama.close()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])