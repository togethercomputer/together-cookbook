<img src="images\together.gif" width="100%"/>

<!-- Links -->
<p align="center">
  <a href="https://api.together.ai/signin" style="color: #06b6d4;">Platform</a> •
  <a href="https://docs.together.ai/docs/introduction" style="color: #06b6d4;">Docs</a> •
  <a href="https://www.together.ai/blog" style="color: #06b6d4;">Blog</a> •
  <a href="https://discord.gg/9Rk6sSeWEG" style="color: #06b6d4;">Discord</a>
</p>

# Together Cookbook

The Together Cookbook is a collection of code and guides designed to help developers build with open source models using [Together AI](https://www.together.ai/). The best way to use the recipes is to copy code snippets and integrate them into your own projects!

## Contributing

We welcome contributions to this repository! If you have a cookbook you'd like to add, please reach out to use the [Discord](https://discord.gg/9Rk6sSeWEG) or open a pull request!

## Prerequisites

To make the most of the examples in this cookbook, you'll need a Together AI API key (sign up for free [here](https://api.together.ai/signin)).

While the code examples are primarily written in Python/JS, the concepts can be adapted to any programming language that supports interaction with the Together API.

## Cookbooks

| Cookbook | Description | Open |
| -------- | ----------- | ---- |
| Agents | | |
| [Serial Chain Agent ](https://github.com/togethercomputer/together-cookbook/blob/main/Agents/Serial_Chain_Agent_Workflow.ipynb) | Chain multiple LLM calls sequentially to process complex tasks. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Serial_Chain_Agent_Workflow.ipynb) |
| [Conditional Router Agent Workflow ](https://github.com/togethercomputer/together-cookbook/blob/main/Agents/Conditional_Router_Agent_Workflow.ipynb) | Create an agent that routes tasks to specialized models. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Conditional_Router_Agent_Workflow.ipynb) |
| [Parallel Agent Workflow ](https://github.com/togethercomputer/together-cookbook/blob/main/Agents/Parallel_Agent_Workflow.ipynb) | Run multiple LLMs in parallel and aggregate their solutions. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Parallel_Agent_Workflow.ipynb) |
| [Orchestrator Subtask Agent Workflow ](https://github.com/togethercomputer/together-cookbook/blob/main/Agents/Parallel_Subtask_Agent_Workflow.ipynb) | Break down tasks into parallel subtasks to be executed by LLMs. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Parallel_Subtask_Agent_Workflow.ipynb) |
| [Looping Agent Workflow](https://github.com/togethercomputer/together-cookbook/blob/main/Agents/Looping_Agent_Workflow.ipynb) | Build an agent that iteratively improves responses | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Looping_Agent_Workflow.ipynb) |
| Fine-tuning | | |
| [LoRA Inference and Fine-tuning](https://github.com/togethercomputer/together-cookbook/blob/main/LoRA_Finetuning%26Inference.ipynb) | Perform LoRA fine-tuning and inference on Together AI. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/LoRA_Finetuning%26Inference.ipynb) [![](https://uohmivykqgnnbiouffke.supabase.co/storage/v1/object/public/landingpage/youtubebadge.svg)](https://youtu.be/AKO7YDcwhiQ?si=ZcOGdDeTRelwIqXd) |
| [Long Context Finetuning For Repetition](https://github.com/togethercomputer/together-cookbook/blob/main/LongContext_Finetuning_RepetitionTask.ipynb) | Fine-tuning LLMs to repeat back words in long sequences. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/LongContext_Finetuning_RepetitionTask.ipynb) |
| [Summarization Long Context Finetuning](https://github.com/togethercomputer/together-cookbook/blob/main/Summarization_LongContext_Finetuning.ipynb) | Long context fine-tuning to improve summarization capabilities. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Summarization_LongContext_Finetuning) |
| [Conversation Finetuning](https://github.com/togethercomputer/together-cookbook/blob/main/Multiturn_Conversation_Finetuning.ipynb) | Fine-tuning LLMs on multi-step conversations. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Multiturn_Conversation_Finetuning.ipynb) |
| Retrieval-augmented generation  |  | |
| [RAG_with_Reasoning_Models](https://github.com/togethercomputer/together-cookbook/blob/main/RAG_with_Reasoning_Models.ipynb) | RAG + source citations with DeepSeek R1. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/RAG_with_Reasoning_Models.ipynb) |
| [MultiModal_RAG_with_Nvidia_Deck](https://github.com/togethercomputer/together-cookbook/blob/main/MultiModal_RAG_with_Nvidia_Investor_Slide_Deck.ipynb) | Multimodal RAG using Nvidia investor slides | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/MultiModal_RAG_with_Nvidia_Investor_Slide_Deck.ipynb) [![](https://uohmivykqgnnbiouffke.supabase.co/storage/v1/object/public/landingpage/youtubebadge.svg)](https://youtu.be/IluARWPYAUc?si=gG90hqpboQgNOAYG)|
| [Open_Contextual_RAG](https://github.com/togethercomputer/together-cookbook/blob/main/Open_Contextual_RAG.ipynb) | An implementation of Contextual Retrieval using open models. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Open_Contextual_RAG.ipynb) |
| [Text_RAG](https://github.com/togethercomputer/together-cookbook/blob/main/Text_RAG.ipynb) | Implement text-based Retrieval-Augmented Generation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google/github.com/togethercomputer/together-cookbook/blob/main/Text_RAG.ipynb) |
| Search  |  |  |
| [Multimodal Search and Conditional Image Generation](https://github.com/togethercomputer/together-cookbook/blob/main/Multimodal_Search_and_Conditional_Image_Generation.ipynb) | Text-to-image and image-to-image search and condtional image generation. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Multimodal_Search_and_Conditional_Image_Generation.ipynb) |
| [Embedding_Visualization](https://github.com/togethercomputer/together-cookbook/blob/main/Embedding_Visualization.ipynb) | Visualize vector embeddings | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Embedding_Visualization.ipynb) |
| [Search_with_Reranking](https://github.com/togethercomputer/together-cookbook/blob/main/Search_with_Reranking.ipynb) | Improve search results with rerankers | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Search_with_Reranking.ipynb) |
| [Semantic_Search](https://github.com/togethercomputer/together-cookbook/blob/main/Semantic_Search.ipynb) | Implement vector search with embedding models | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Semantic_Search.ipynb) |
| Miscellaneous | | |
| [Thinking_Augmented_Generation](https://github.com/togethercomputer/together-cookbook/blob/main/Thinking_Augmented_Generation.ipynb) | Give R1 thinking tokens to small models | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Thinking_Augmented_Generation.ipynb) |
| [Flux LoRA Inference](https://github.com/togethercomputer/together-cookbook/blob/main/Flux_LoRA_Inference.ipynb) | Generate images with fine-tuned Flux LoRA's | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Flux_LoRA_Inference.ipynb) |
| [Structured_Text_Extraction_from_Images](https://github.com/togethercomputer/together-cookbook/blob/main/Structured_Text_Extraction_from_Images.ipynb) | Extract structured text from images | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Structured_Text_Extraction_from_Images.ipynb) |
| [Summarization Evaluation](https://github.com/togethercomputer/together-cookbook/blob/main/Summarization_Evaluation.ipynb) | Summarizing and evaluating outputs with LLMs. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Summarization_Evaluation.ipynb) |
| [PDF_to_Podcast](https://github.com/togethercomputer/together-cookbook/blob/main/PDF_to_Podcast.ipynb) | Generate a podcast from PDF content NotebookLM style! | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/PDF_to_Podcast.ipynb) |
| [Knowledge_Graphs_with_Structured_Outputs](https://github.com/togethercomputer/together-cookbook/blob/main/Knowledge_Graphs_with_Structured_Outputs.ipynb) | Get LLMs to generate knowledge graphs | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Knowledge_Graphs_with_Structured_Outputs.ipynb) |

## Explore Further

Looking for more resources to enhance your experience with open source models? Check out these helpful links:

- [Together AI developer documentation](https://docs.together.ai/docs/introduction)
- [Together AI API support docs](https://docs.together.ai/reference/chat-completions-1)
- [Together AI Discord community](https://discord.gg/9Rk6sSeWEG)

## Additional Resources

- [Together AI Research](https://www.together.ai/research): Explore papers and technical blog posts from our research team.
- [Together AI Blog](https://www.together.ai/blog): Explore technical blogs, product announcements and more on our blog.
