<img src="images/together.gif" width="100%"/>

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

While the code examples are primarily written in Python, the concepts can be adapted to any programming language that supports interaction with the Together API.

---

## 🚀 [Quickstarts](Quickstarts/)

Get started with Together AI in minutes

| Notebook | Description | Open |
| -------- | ----------- | ---- |
| [Getting Started with Llama 4](Quickstarts/Getting_started_with_Llama4.ipynb) | Learn to use Llama 4 Maverick and Scout models | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Quickstarts/Getting_started_with_Llama4.ipynb) |
| [Thinking-Augmented Generation](Quickstarts/Thinking_Augmented_Generation.ipynb) | Give R1 thinking tokens to small models | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Quickstarts/Thinking_Augmented_Generation.ipynb) |
| [Knowledge Graphs](Quickstarts/Knowledge_Graphs_with_Structured_Outputs.ipynb) | Generate knowledge graphs with LLMs | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Quickstarts/Knowledge_Graphs_with_Structured_Outputs.ipynb) |

---

## 🔍 [RAG and Search](RAG_and_Search/)

Build intelligent retrieval systems with semantic search and retrieval-augmented generation

| Notebook | Description | Open |
| -------- | ----------- | ---- |
| [Text RAG](RAG_and_Search/Text_RAG.ipynb) | Basic RAG implementation with vector embeddings | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/RAG_and_Search/Text_RAG.ipynb) |
| [Semantic Search](RAG_and_Search/Semantic_Search.ipynb) | Implement vector search with embedding models | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/RAG_and_Search/Semantic_Search.ipynb) |
| [Search with Reranking](RAG_and_Search/Search_with_Reranking.ipynb) | Improve search results with rerankers | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/RAG_and_Search/Search_with_Reranking.ipynb) |
| [Open Contextual RAG](RAG_and_Search/Open_Contextual_RAG.ipynb) | Contextual Retrieval using open models | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/RAG_and_Search/Open_Contextual_RAG.ipynb) |
| [RAG with Reasoning Models](RAG_and_Search/RAG_with_Reasoning_Models.ipynb) | RAG + citations with DeepSeek R1 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/RAG_and_Search/RAG_with_Reasoning_Models.ipynb) |
| [Embedding Visualization](RAG_and_Search/Embedding_Visualization.ipynb) | Visualize vector embeddings | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/RAG_and_Search/Embedding_Visualization.ipynb) |
| [Contextual RAG on Union](RAG_and_Search/contextual_rag_on_union/) | Full-stack RAG with FastAPI + Gradio | - |

---

## 🤖 [Agents](Agents/)

Build autonomous AI workflows with fundamental agent patterns

**Core Workflows:**
| Workflow | Description | Open |
| -------- | ----------- | ---- |
| [Serial Chain Agent](Agents/Serial_Chain_Agent_Workflow.ipynb) | Chain multiple LLM calls sequentially | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Serial_Chain_Agent_Workflow.ipynb) |
| [Conditional Router Agent](Agents/Conditional_Router_Agent_Workflow.ipynb) | Route tasks to specialized models | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Conditional_Router_Agent_Workflow.ipynb) |
| [Parallel Agent Workflow](Agents/Parallel_Agent_Workflow.ipynb) | Run multiple LLMs in parallel | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Parallel_Agent_Workflow.ipynb) |
| [Orchestrator Subtask Agent](Agents/Parallel_Subtask_Agent_Workflow.ipynb) | Break down tasks into parallel subtasks | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Parallel_Subtask_Agent_Workflow.ipynb) |
| [Looping Agent Workflow](Agents/Looping_Agent_Workflow.ipynb) | Iteratively improve responses | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Looping_Agent_Workflow.ipynb) |

**Specialized Agents:**
| Agent | Description | Open |
| ----- | ----------- | ---- |
| [Together Open Deep Research](Agents/Together_Open_Deep_Research_CookBook.ipynb) | Multi-step web search agent | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Together_Open_Deep_Research_CookBook.ipynb) |
| [Data Science Agent](Agents/DataScienceAgent/Together_Open_DataScience_Agent.ipynb) | ReAct agent for data analysis | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/DataScienceAgent/Together_Open_DataScience_Agent.ipynb) |

---

## 🎛️ [Fine-tuning](Finetuning/)

Customize models for your specific needs

| Notebook | Description | Open |
| -------- | ----------- | ---- |
| [End-to-end Fine-tuning Guide](Finetuning/Finetuning_Guide.ipynb) | Fine-tuning basics and best practices | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Finetuning/Finetuning_Guide.ipynb) [![YouTube](https://img.shields.io/badge/YouTube-Video-red)](https://youtu.be/AKO7YDcwhiQ?si=ZcOGdDeTRelwIqXd) |
| [LoRA Inference and Fine-tuning](Finetuning/LoRA_Finetuning_and_Inference.ipynb) | Parameter-efficient fine-tuning | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Finetuning/LoRA_Finetuning_and_Inference.ipynb) |
| [Preference Tuning - DPO](Finetuning/DPO_Finetuning.ipynb) | Fine-tune with preference data using DPO | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Finetuning/DPO_Finetuning.ipynb) |
| [Continual Fine-tuning](Finetuning/Continual_Finetuning.ipynb) | Continuously fine-tune on new data | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Finetuning/Continual_Finetuning.ipynb) |
| [Long Context Finetuning](Finetuning/LongContext_Finetuning_RepetitionTask.ipynb) | Fine-tune for long sequences | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Finetuning/LongContext_Finetuning_RepetitionTask.ipynb) |
| [Summarization Long Context](Finetuning/Summarization_LongContext_Finetuning.ipynb) | Improve summarization capabilities | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Finetuning/Summarization_LongContext_Finetuning.ipynb) |
| [Conversation Finetuning](Finetuning/Multiturn_Conversation_Finetuning.ipynb) | Fine-tune on multi-step conversations | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Finetuning/Multiturn_Conversation_Finetuning.ipynb) |
| [Flux LoRA Inference](Finetuning/Flux_LoRA_Inference.ipynb) | Generate images with fine-tuned Flux LoRAs | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Finetuning/Flux_LoRA_Inference.ipynb) |

---

## 📊 [Evals](Evals/)

Evaluate and benchmark model performance

| Notebook | Description | Open |
| -------- | ----------- | ---- |
| [Classification Evals](Evals/Classification_Evals.ipynb) | LLM-as-a-Judge for safety evaluation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Evals/Classification_Evals.ipynb) |
| [Compare Evals](Evals/Compare_Evals.ipynb) | Head-to-head model comparison | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Evals/Compare_Evals.ipynb) |
| [Prompt Evals](Evals/Prompt_Evals.ipynb) | Prompt optimization through A/B testing | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Evals/Prompt_Evals.ipynb) |
| [Batch Inference Evals](Evals/Batch_Inference_Evals.ipynb) | Batch inference and evaluation workflows | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Evals/Batch_Inference_Evals.ipynb) |
| [Summarization Evaluation](Evals/Summarization_Evaluation.ipynb) | Evaluate summarization outputs | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Evals/Summarization_Evaluation.ipynb) |

---

## 🛠️ [Tools and Integrations](Tools_and_Integrations/)

Extend capabilities with code execution, tool calling, and agent frameworks

**Code Execution:**
| Tool | Description | Open |
| ---- | ----------- | ---- |
| [Together Code Interpreter](Tools_and_Integrations/Code_Execution/Together_Code_Interpreter.ipynb) | Execute code using Together Code Interpreter | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Tools_and_Integrations/Code_Execution/Together_Code_Interpreter.ipynb) |
| [OpenEnv Code Interpreter](Tools_and_Integrations/Code_Execution/OpenEnv_Code_Interpreter/) | Code Interpreter as OpenEnv environment | - |

**Tool Calling:**
| Tool | Description | Open |
| ---- | ----------- | ---- |
| [Tool Use with Toolhouse](Tools_and_Integrations/Tool_Calling/Tool_use_with_Toolhouse.ipynb) | Function calling with Toolhouse toolkit | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Tools_and_Integrations/Tool_Calling/Tool_use_with_Toolhouse.ipynb) |

**Agent Frameworks:**
| Framework | Notebooks |
| --------- | --------- |
| [LangGraph](Tools_and_Integrations/Agent_Frameworks/LangGraph/) | Agentic RAG, Planning Agent |
| [DSPy](Tools_and_Integrations/Agent_Frameworks/DSPy/) | Agents with MIPRO optimization |
| [Agno](Tools_and_Integrations/Agent_Frameworks/Agno/), [Composio](Tools_and_Integrations/Agent_Frameworks/Composio/), [Arcade](Tools_and_Integrations/Agent_Frameworks/Arcade/), [KlavisAI](Tools_and_Integrations/Agent_Frameworks/KlavisAI/), [PydanticAI](Tools_and_Integrations/Agent_Frameworks/PydanticAI/) | Various framework integrations |

---

## ☁️ [Together Instant Clusters](Together_Instant_Clusters/)

Scale with distributed training on Kubernetes

| Example | Description |
| ------- | ----------- |
| [GRPO BlackJack](Together_Instant_Clusters/GRPO_BlackJack/) | Train Qwen 1.5B with GRPO on Together Instant Clusters |

---

## 🎨 [Multimodal](Multimodal/)

Work with images, audio, and cross-modal AI tasks

| Notebook | Description | Open |
| -------- | ----------- | ---- |
| [MultiModal RAG with Nvidia Deck](Multimodal/MultiModal_RAG_with_Nvidia_Investor_Slide_Deck.ipynb) | RAG over slides using ColPali | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Multimodal/MultiModal_RAG_with_Nvidia_Investor_Slide_Deck.ipynb) [![YouTube](https://img.shields.io/badge/YouTube-Video-red)](https://youtu.be/IluARWPYAUc?si=gG90hqpboQgNOAYG) |
| [Multimodal Search & Image Gen](Multimodal/Multimodal_Search_and_Conditional_Image_Generation.ipynb) | Text-to-image and image-to-image search | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Multimodal/Multimodal_Search_and_Conditional_Image_Generation.ipynb) |
| [Structured Text from Images](Multimodal/Structured_Text_Extraction_from_Images.ipynb) | Extract structured data from images | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Multimodal/Structured_Text_Extraction_from_Images.ipynb) |
| [PDF to Podcast](Multimodal/PDF_to_Podcast.ipynb) | Generate podcasts from PDFs | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Multimodal/PDF_to_Podcast.ipynb) |

---

## Explore Further

Looking for more resources to enhance your experience with open source models? Check out these helpful links:

- [Together AI developer documentation](https://docs.together.ai/docs/introduction)
- [Together AI API support docs](https://docs.together.ai/reference/chat-completions-1)
- [Together AI Discord community](https://discord.gg/9Rk6sSeWEG)

## Additional Resources

- [Together AI Research](https://www.together.ai/research): Explore papers and technical blog posts from our research team.
- [Together AI Blog](https://www.together.ai/blog): Explore technical blogs, product announcements and more on our blog.
