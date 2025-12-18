"""
Together AI Client Utilities for Qwen3-VL

This module provides shared utilities for calling Together AI's API
with the Qwen3-VL vision-language model.

Usage:
    export TOGETHER_API_KEY=your_key_here

    from utils.together_client import inference_with_image, inference_with_images
"""

import os
import base64
import openai
from PIL import Image
from io import BytesIO

# Together AI configuration
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_MODEL = "Qwen/Qwen3-VL-32B-Instruct"


def get_client():
    """
    Get OpenAI client configured for Together AI.
    
    Returns:
        openai.OpenAI: Configured client instance
    """
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError(
            "TOGETHER_API_KEY environment variable not set. "
            "Get your API key from https://api.together.xyz/settings/api-keys"
        )
    return openai.OpenAI(
        api_key=api_key,
        base_url=TOGETHER_BASE_URL,
    )


def encode_image(image_path):
    """
    Encode a local image file to base64.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pil_to_base64(pil_image, format="PNG"):
    """
    Convert PIL Image to base64 string.
    
    Args:
        pil_image: PIL Image object
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        str: Base64 encoded image string
    """
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_mime_type(image_path):
    """Get MIME type from file extension."""
    ext = image_path.split(".")[-1].lower()
    mime_map = {
        "jpg": "jpeg",
        "jpeg": "jpeg", 
        "png": "png",
        "gif": "gif",
        "webp": "webp"
    }
    return mime_map.get(ext, "jpeg")


def inference_with_image(image_path_or_url, prompt, max_tokens=4096, temperature=None):
    """
    Run inference with a single image.
    
    Args:
        image_path_or_url: Local path or URL to image
        prompt: Text prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (optional)
    
    Returns:
        str: Model response text
    """
    client = get_client()
    
    # Handle local file vs URL
    if image_path_or_url.startswith(("http://", "https://")):
        image_content = {
            "type": "image_url", 
            "image_url": {"url": image_path_or_url}
        }
    else:
        base64_img = encode_image(image_path_or_url)
        mime_type = get_mime_type(image_path_or_url)
        image_content = {
            "type": "image_url", 
            "image_url": {"url": f"data:image/{mime_type};base64,{base64_img}"}
        }
    
    kwargs = {
        "model": TOGETHER_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                image_content
            ]
        }],
        "max_tokens": max_tokens
    }
    
    if temperature is not None:
        kwargs["temperature"] = temperature
    
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def inference_with_images(images, prompt, max_tokens=250_000, temperature=None):
    """
    Run inference with multiple images (e.g., for PDF pages).
    
    Args:
        images: List of PIL Images, file paths, or URLs
        prompt: Text prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (optional)
    
    Returns:
        str: Model response text
    """
    client = get_client()
    
    content = []
    for img in images:
        if isinstance(img, str):
            if img.startswith(("http://", "https://")):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img}
                })
            else:
                base64_img = encode_image(img)
                mime_type = get_mime_type(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime_type};base64,{base64_img}"}
                })
        else:  # PIL Image
            base64_img = pil_to_base64(img, format="PNG")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}"}
            })
    
    content.append({"type": "text", "text": prompt})
    
    kwargs = {
        "model": TOGETHER_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens
    }
    
    if temperature is not None:
        kwargs["temperature"] = temperature
    
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def inference_with_video(video_url, prompt, max_tokens=4096, temperature=None):
    """
    Run inference with a video URL.
    
    Note: Together AI only supports video URLs, not local files or frame lists.
    
    Args:
        video_url: URL to video file (must be publicly accessible)
        prompt: Text prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (optional)
    
    Returns:
        str: Model response text
    """
    client = get_client()
    
    kwargs = {
        "model": TOGETHER_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video_url", "video_url": {"url": video_url}}
            ]
        }],
        "max_tokens": max_tokens
    }
    
    if temperature is not None:
        kwargs["temperature"] = temperature
    
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def inference_with_system_prompt(image_path_or_url, prompt, system_prompt, max_tokens=4096, temperature=None):
    """
    Run inference with a system prompt.
    
    Args:
        image_path_or_url: Local path or URL to image
        prompt: User text prompt
        system_prompt: System prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (optional)
    
    Returns:
        str: Model response text
    """
    client = get_client()
    
    # Handle local file vs URL
    if image_path_or_url.startswith(("http://", "https://")):
        image_content = {
            "type": "image_url", 
            "image_url": {"url": image_path_or_url}
        }
    else:
        base64_img = encode_image(image_path_or_url)
        mime_type = get_mime_type(image_path_or_url)
        image_content = {
            "type": "image_url", 
            "image_url": {"url": f"data:image/{mime_type};base64,{base64_img}"}
        }
    
    kwargs = {
        "model": TOGETHER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content
                ]
            }
        ],
        "max_tokens": max_tokens
    }
    
    if temperature is not None:
        kwargs["temperature"] = temperature
    
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

