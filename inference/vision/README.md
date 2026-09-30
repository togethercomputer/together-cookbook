# Vision with Qwen3.5

Explore Qwen3.5's vision-language capabilities using Together AI's API. From OCR to 3D grounding, these notebooks cover a wide range of visual understanding tasks. These notebooks are adapted from the original Qwen3-VL [repo](https://github.com/QwenLM/Qwen3-VL/tree/main).

## 📚 Notebooks

| Notebook | Description | Open |
| -------- | ----------- | ---- |
| [OCR](OCR.ipynb) | Text extraction, multilingual OCR, and text spotting with bounding boxes | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/inference/vision/OCR.ipynb) |
| [2D Grounding](2D_Grounding.ipynb) | Object detection with 2D bounding boxes, multi-target detection, point grounding | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/inference/vision/2D_Grounding.ipynb) |
| [3D Grounding](3D_Grounding.ipynb) | 3D bounding boxes, camera parameters, depth perception | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/inference/vision/3D_Grounding.ipynb) |
| [Spatial Understanding](Spatial_Understanding.ipynb) | Object relationships, affordances, embodied reasoning | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/inference/vision/Spatial_Understanding.ipynb) |
| [Video Understanding](Video_Understanding.ipynb) | Video description, temporal localization, video Q&A | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/inference/vision/Video_Understanding.ipynb) |
| [Omni Recognition](Omni_Recognition.ipynb) | Universal recognition for celebrities, anime, food, landmarks | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/inference/vision/Omni_Recognition.ipynb) |
| [Document Parsing](Document_Parsing.ipynb) | Convert documents to HTML/Markdown with coordinates | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/inference/vision/Document_Parsing.ipynb) |
| [Image to Code](Image_to_Code.ipynb) | Screenshot to HTML, chart to matplotlib code | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/inference/vision/Image_to_Code.ipynb) |
| [Long Document Understanding](Long_Document_Understanding.ipynb) | Multi-page PDF analysis and Q&A | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/inference/vision/Long_Document_Understanding.ipynb) |

## 🎯 Key Concepts

### Coordinate System
Qwen3-VL uses a **relative coordinate system from 0 to 1000**:
- `bbox_2d`: `[x1, y1, x2, y2]` - top-left and bottom-right corners
- `point_2d`: `[x, y]` - point coordinates
- `bbox_3d`: `[x, y, z, x_size, y_size, z_size, roll, pitch, yaw]` - 3D box

### Token Calculation
Together AI calculates image tokens as:
```
T = min(2, max(H // 560, 1)) * min(2, max(W // 560, 1)) * 1601
```
- Per image: ~1,601 to 6,404 tokens
- Plan accordingly for multi-image inputs

## 🚀 Quick Start

```python
import os
import together

client = together.Together()

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-397B-A17B",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
            {"type": "text", "text": "What's in this image?"},
        ],
    }],
)
print(response.choices[0].message.content)
```

## 📋 Prerequisites

- Together AI API key ([get one here](https://api.together.ai/settings/api-keys))
- Python 3.8+
- Additional dependencies per notebook (see each notebook for details)

## 📖 Resources

- [Together AI Documentation](https://docs.together.ai/intro)
- [Together AI Vision Guide](https://docs.together.ai/docs/vision-overview)
- [Qwen3.5 397B Model Card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
- [Qwen3-VL Reference (original)](https://github.com/QwenLM/Qwen3-VL)
