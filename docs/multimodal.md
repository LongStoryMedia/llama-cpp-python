# Multimodal Support for Qwen3-VL

This document describes how to use multimodal (vision-language) models with `llama-cpp-python`, specifically focusing on Qwen3-VL models.

## Overview

Qwen3-VL models combine large language model capabilities with computer vision, allowing them to understand and reason about images alongside text. This implementation supports:

- Loading Qwen3-VL models with multimodal projectors (mmproj)
- Processing images from URLs or base64 data
- Chat completion with mixed text and image inputs
- Proper formatting for Qwen3-VL's chat template

## Model Requirements

To use multimodal functionality, you need two files:

1. **Main model file**: The primary GGUF model file (e.g., `ggml-model-f16.gguf`)
2. **Multimodal projector file**: The vision encoder GGUF file (e.g., `mmproj-model-f16.gguf`)

### Downloading Models

You can download pre-converted Qwen3-VL models from Hugging Face:

```bash
# Example: Qwen3-VL-2B model
wget https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-2B-Thinking-abliterated/resolve/main/GGUF/ggml-model-f16.gguf
wget https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-2B-Thinking-abliterated/resolve/main/GGUF/mmproj-model-f16.gguf
```

## Usage

### Basic Setup

```python
import llama_cpp
from llama_cpp import llama_chat_format

# Create the multimodal chat handler
chat_handler = llama_chat_format.Qwen25VLChatHandler(
    clip_model_path="path/to/mmproj-model-f16.gguf",
    verbose=True
)

# Load the model with multimodal support
llama = llama_cpp.Llama(
    model_path="path/to/ggml-model-f16.gguf",
    chat_handler=chat_handler,
    n_ctx=4096,  # Larger context for multimodal tasks
    n_gpu_layers=-1,  # Use GPU if available
    verbose=True
)
```

### Text-only Conversation

```python
messages = [
    {"role": "user", "content": "Hello! What can you do?"}
]

response = llama.create_chat_completion(
    messages=messages,
    max_tokens=200,
    temperature=0.7
)

print(response['choices'][0]['message']['content'])
```

### Single Image Analysis

```python
messages = [
    {
        "role": "user", 
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"}
            },
            {
                "type": "text", 
                "text": "What do you see in this image?"
            }
        ]
    }
]

response = llama.create_chat_completion(
    messages=messages,
    max_tokens=300,
    temperature=0.3
)

print(response['choices'][0]['message']['content'])
```

### Multiple Image Comparison

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image1.jpg"}
            },
            {
                "type": "image_url", 
                "image_url": {"url": "https://example.com/image2.jpg"}
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
    max_tokens=400,
    temperature=0.4
)

print(response['choices'][0]['message']['content'])
```

### Using Base64 Images

```python
# Load image as base64
import base64
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            },
            {
                "type": "text",
                "text": "Describe this image."
            }
        ]
    }
]

response = llama.create_chat_completion(messages=messages, max_tokens=200)
```

### Streaming Responses

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"}
            },
            {
                "type": "text",
                "text": "Tell me a story about this image."
            }
        ]
    }
]

response = llama.create_chat_completion(
    messages=messages,
    max_tokens=500,
    temperature=0.8,
    stream=True
)

for chunk in response:
    if 'choices' in chunk:
        delta = chunk['choices'][0].get('delta', {})
        if 'content' in delta:
            print(delta['content'], end='', flush=True)
```

## Chat Format

Qwen3-VL uses a specific chat format with vision tokens:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Picture 1: <|vision_start|> image_url <|vision_end|>
What do you see?<|im_end|>
<|im_start|>assistant
```

The `Qwen25VLChatHandler` automatically handles this formatting, including:
- Adding system messages
- Numbering images as "Picture 1", "Picture 2", etc.
- Wrapping image URLs with vision tokens
- Proper conversation flow

## Configuration Options

### Chat Handler Options

```python
chat_handler = llama_chat_format.Qwen25VLChatHandler(
    clip_model_path="path/to/mmproj.gguf",
    verbose=False  # Set to True for debugging
)
```

### Model Options

```python
llama = llama_cpp.Llama(
    model_path="path/to/model.gguf",
    chat_handler=chat_handler,
    n_ctx=4096,        # Context size - larger for complex multimodal tasks
    n_gpu_layers=-1,   # GPU layers - use all available
    n_threads=8,       # CPU threads
    verbose=False      # Logging verbosity
)
```

### Generation Options

```python
response = llama.create_chat_completion(
    messages=messages,
    max_tokens=300,     # Maximum response length
    temperature=0.7,    # Creativity (0.0-1.0)
    top_p=0.9,         # Nucleus sampling
    top_k=40,          # Top-k sampling
    stream=False       # Enable streaming
)
```

## Performance Tips

1. **Context Size**: Use larger context sizes (4096+) for complex multimodal tasks
2. **GPU Usage**: Enable GPU layers with `n_gpu_layers=-1` for faster inference
3. **Image Quality**: Higher resolution images provide more detail but use more memory
4. **Batch Processing**: Process multiple images in a single request when comparing
5. **Temperature**: Use lower temperatures (0.1-0.3) for factual descriptions, higher (0.6-0.9) for creative tasks

## Error Handling

Common issues and solutions:

### Missing Files
```python
try:
    chat_handler = llama_chat_format.Qwen25VLChatHandler(
        clip_model_path="mmproj.gguf"
    )
except ValueError as e:
    print(f"Error: {e}")
    # Check file exists and path is correct
```

### Memory Issues
```python
# Reduce context size for memory-constrained systems
llama = llama_cpp.Llama(
    model_path="model.gguf",
    chat_handler=chat_handler,
    n_ctx=2048,  # Smaller context
    n_gpu_layers=20  # Partial GPU usage
)
```

### Image Loading Errors
```python
try:
    response = llama.create_chat_completion(messages=messages)
except Exception as e:
    print(f"Error processing image: {e}")
    # Check image URL accessibility and format
```

## Testing

Run the multimodal tests to verify your setup:

```bash
# Run all multimodal tests
python tests/test_multimodal_qwen3vl.py

# Run with pytest
python -m pytest tests/test_multimodal_qwen3vl.py -v
```

## Examples

See `examples/multimodal_qwen3vl_example.py` for comprehensive usage examples including:
- Interactive multimodal chat
- Image analysis workflows  
- Multi-image processing
- Error handling patterns

## Supported Image Formats

- JPEG
- PNG
- GIF (static)
- WebP
- URLs (http/https)
- Base64 encoded images
- Local file paths (when converted to base64)

## Limitations

- Maximum image resolution depends on model and available memory
- Processing time increases with image size and number of images
- Some very large or complex images may cause memory issues
- GPU memory usage scales with image resolution and model size

## Troubleshooting

1. **Segmentation faults**: Usually indicate memory issues - try smaller images or context size
2. **Slow processing**: Enable GPU layers and check image resolution
3. **Poor quality responses**: Try different temperature settings or provide more context
4. **Image not loading**: Verify URL accessibility and image format support

For more help, check the test files and example scripts for working code patterns.