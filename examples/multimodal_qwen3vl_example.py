#!/usr/bin/env python3
"""
Qwen3-VL Multimodal Example

This script demonstrates how to use Qwen3-VL models with llama-cpp-python
for vision-language tasks.

Usage:
    python examples/multimodal_qwen3vl_example.py

Requirements:
    - Qwen3-VL GGUF model file
    - Qwen3-VL multimodal projector (mmproj) file
    - Internet connection for downloading test images
"""

import sys
from pathlib import Path

import llama_cpp
from llama_cpp import llama_chat_format


# Example model URLs (adjust these to your actual model paths)
# Download from: https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-2B-Thinking-abliterated
MODEL_PATH = "path/to/your/qwen3-vl-model.gguf"
MMPROJ_PATH = "path/to/your/qwen3-vl-mmproj.gguf"

# Test images
TEST_IMAGES = [
    "https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
]


def setup_multimodal_model(model_path: str, mmproj_path: str) -> llama_cpp.Llama:
    """
    Initialize a Qwen3-VL multimodal model.

    Args:
        model_path: Path to the main GGUF model file
        mmproj_path: Path to the multimodal projector GGUF file

    Returns:
        Configured Llama instance with multimodal support
    """
    print("Setting up Qwen3-VL multimodal model...")

    # Check if files exist
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not Path(mmproj_path).exists():
        raise FileNotFoundError(f"Multimodal projector file not found: {mmproj_path}")

    # Create the multimodal chat handler
    chat_handler = llama_chat_format.Qwen25VLChatHandler(
        clip_model_path=mmproj_path, verbose=True
    )

    # Initialize the model with multimodal support
    llama = llama_cpp.Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=4096,  # Large context for multimodal tasks
        n_gpu_layers=-1,  # Use GPU if available
        verbose=True,
    )

    print("✓ Multimodal model loaded successfully!")
    return llama


def example_text_only_conversation(llama: llama_cpp.Llama):
    """Example of text-only conversation with multimodal model."""
    print("\n" + "=" * 50)
    print("Example 1: Text-only conversation")
    print("=" * 50)

    messages = [
        {"role": "user", "content": "Hello! Can you tell me about your capabilities?"}
    ]

    print("User: Hello! Can you tell me about your capabilities?")
    print("Assistant: ", end="")

    response = llama.create_chat_completion(
        messages=messages,  # type: ignore
        max_tokens=200,
        temperature=0.7,
        stream=True,  # Stream the response for real-time output
    )

    # Handle streaming response
    for chunk in response:
        if isinstance(chunk, dict) and "choices" in chunk:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                print(delta["content"], end="", flush=True)

    print("\n")


def example_single_image_analysis(llama: llama_cpp.Llama, image_url: str):
    """Example of analyzing a single image."""
    print("\n" + "=" * 50)
    print("Example 2: Single image analysis")
    print("=" * 50)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {
                    "type": "text",
                    "text": "Please describe this image in detail. What do you see?",
                },
            ],
        }
    ]

    print(f"Analyzing image: {image_url}")
    print("User: Please describe this image in detail. What do you see?")
    print("Assistant: ", end="")

    response = llama.create_chat_completion(
        messages=messages,  # type: ignore
        max_tokens=300,
        temperature=0.3,  # Lower temperature for more focused descriptions
        stream=True,
    )

    # Handle streaming response
    for chunk in response:
        if isinstance(chunk, dict) and "choices" in chunk:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                print(delta["content"], end="", flush=True)

    print("\n")


def example_multi_image_comparison(llama: llama_cpp.Llama, image_urls: list[str]):
    """Example of comparing multiple images."""
    print("\n" + "=" * 50)
    print("Example 3: Multi-image comparison")
    print("=" * 50)

    # Build message content with multiple images
    content = []

    for i, url in enumerate(image_urls, 1):
        content.append({"type": "image_url", "image_url": {"url": url}})

    content.append(
        {
            "type": "text",
            "text": "Compare these images. What are the similarities and differences? What can you tell me about each image?",
        }
    )

    messages = [{"role": "user", "content": content}]

    print(f"Comparing {len(image_urls)} images:")
    for i, url in enumerate(image_urls, 1):
        print(f"  Image {i}: {url}")

    print("User: Compare these images. What are the similarities and differences?")
    print("Assistant: ", end="")

    response = llama.create_chat_completion(
        messages=messages, max_tokens=500, temperature=0.4, stream=True  # type: ignore
    )

    # Handle streaming response
    for chunk in response:
        if isinstance(chunk, dict) and "choices" in chunk:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                print(delta["content"], end="", flush=True)

    print("\n")


def example_interactive_conversation(llama: llama_cpp.Llama):
    """Example of interactive multimodal conversation."""
    print("\n" + "=" * 50)
    print("Example 4: Interactive conversation")
    print("Type 'quit' to exit, 'help' for commands")
    print("=" * 50)

    messages = []

    while True:
        try:
            user_input = input("\nUser: ").strip()

            if user_input.lower() == "quit":
                break
            elif user_input.lower() == "help":
                print("Commands:")
                print("  quit - Exit the conversation")
                print("  help - Show this help")
                print("  To add an image: [IMAGE:url] followed by your question")
                print(
                    "  Example: [IMAGE:https://example.com/image.jpg] What's in this image?"
                )
                continue
            elif not user_input:
                continue

            # Parse image URLs from user input
            content = []
            remaining_text = user_input

            # Simple parsing for [IMAGE:url] syntax
            while "[IMAGE:" in remaining_text:
                start = remaining_text.find("[IMAGE:")
                if start == -1:
                    break
                end = remaining_text.find("]", start)
                if end == -1:
                    break

                # Extract image URL
                image_url = remaining_text[start + 7 : end]
                content.append({"type": "image_url", "image_url": {"url": image_url}})

                # Remove the image tag from text
                remaining_text = remaining_text[:start] + remaining_text[end + 1 :]

            # Add remaining text
            if remaining_text.strip():
                content.append({"type": "text", "text": remaining_text.strip()})

            # If no structured content, treat as simple text
            if not content:
                content = user_input

            messages.append({"role": "user", "content": content})

            print("Assistant: ", end="")

            response = llama.create_chat_completion(
                messages=messages,  # type: ignore
                max_tokens=300,
                temperature=0.6,
                stream=True,
            )

            assistant_response = ""
            for chunk in response:
                if isinstance(chunk, dict) and "choices" in chunk:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        content_part = delta["content"]
                        print(content_part, end="", flush=True)
                        assistant_response += content_part

            print()  # New line after response

            # Add assistant response to conversation history
            messages.append({"role": "assistant", "content": assistant_response})

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


def main():
    """Main example function."""
    print("Qwen3-VL Multimodal Example")
    print("=" * 50)

    # Check if model paths are configured
    if MODEL_PATH == "path/to/your/qwen3-vl-model.gguf":
        print(
            "ERROR: Please update MODEL_PATH and MMPROJ_PATH variables in this script"
        )
        print("       with the actual paths to your Qwen3-VL model files.")
        print()
        print("You can download models from:")
        print(
            "https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-2B-Thinking-abliterated"
        )
        return 1

    try:
        # Setup the multimodal model
        llama = setup_multimodal_model(MODEL_PATH, MMPROJ_PATH)

        # Run examples
        example_text_only_conversation(llama)

        if len(TEST_IMAGES) >= 1:
            example_single_image_analysis(llama, TEST_IMAGES[0])

        if len(TEST_IMAGES) >= 2:
            example_multi_image_comparison(llama, TEST_IMAGES[:2])

        # Interactive example (optional)
        response = input("\nWould you like to try interactive mode? (y/N): ")
        if response.lower().startswith("y"):
            example_interactive_conversation(llama)

        # Cleanup
        llama.close()
        print("\n✓ Example completed successfully!")
        return 0

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(
            "Please make sure the model files exist and update the paths in this script."
        )
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
