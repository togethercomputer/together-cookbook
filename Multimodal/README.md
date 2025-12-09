# Multimodal

Work with images, audio, video, and cross-modal AI tasks using open-source models.

## 📚 Notebooks

| Notebook | Description | Open |
| -------- | ----------- | ---- |
| [MultiModal RAG with Nvidia Investor Slide Deck](MultiModal_RAG_with_Nvidia_Investor_Slide_Deck.ipynb) | RAG over presentation slides using ColPali for visual understanding | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Multimodal/MultiModal_RAG_with_Nvidia_Investor_Slide_Deck.ipynb) [![YouTube](https://img.shields.io/badge/YouTube-Video-red)](https://youtu.be/IluARWPYAUc?si=gG90hqpboQgNOAYG) |
| [Multimodal Search and Conditional Image Generation](Multimodal_Search_and_Conditional_Image_Generation.ipynb) | Text-to-image and image-to-image search with CLIP, plus conditional image generation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Multimodal/Multimodal_Search_and_Conditional_Image_Generation.ipynb) |
| [Structured Text Extraction from Images](Structured_Text_Extraction_from_Images.ipynb) | Extract structured information from images using vision-language models | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Multimodal/Structured_Text_Extraction_from_Images.ipynb) |
| [PDF to Podcast](PDF_to_Podcast.ipynb) | Transform PDF documents into engaging podcast conversations (NotebookLM style) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Multimodal/PDF_to_Podcast.ipynb) |

## 🎯 Key Capabilities

### Vision Understanding
- **Document Analysis**: Extract text and structure from images/PDFs
- **Visual Question Answering**: Ask questions about images
- **Image Search**: Find similar images using CLIP embeddings
- **OCR & Layout**: Understand document structure and extract information

### Audio Generation
- **Text-to-Speech**: Convert text to natural-sounding audio
- **Multi-speaker Conversations**: Generate podcast-style dialogues
- **Voice Cloning**: Custom voices for audio generation

### Cross-Modal Tasks
- **Visual RAG**: Retrieval over images and documents
- **Image-conditioned Generation**: Create images based on reference images
- **Document-to-Audio**: Transform written content to audio format

## 🔧 Models Used

- **Vision-Language Models**: Llama 4 Scout, CLIP
- **Image Understanding**: ColPali (for visual document retrieval)
- **Text-to-Speech**: Cartesia, other TTS models
- **Image Generation**: Flux, Stable Diffusion variants

## 📊 Common Patterns

1. **Multimodal RAG**: Embed images/docs → Vector search → VLM generation
2. **Visual Search**: CLIP embeddings → Similarity search → Results
3. **Document Processing**: OCR/VLM extraction → Structured output
4. **Content Transformation**: PDF → Text → LLM script → TTS audio

## Prerequisites

- Together AI API key
- Understanding of embeddings and vector search
- Basic knowledge of vision-language models
- For audio: TTS service API key (e.g., Cartesia)
