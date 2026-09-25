# DX-1042 static triage of cookbook notebooks (2026-09-15)

This is the **one-time static triage input** for the execution-CI rollout
([DX-1042](https://linear.app/together-ai/issue/DX-1042)). It was produced by
scanning notebook sources only. **No notebook results here are live-confirmed**
except where noted; the full live triage runs once the CI `TOGETHER_API_KEY`
is funded and DevRel signs off on the workflow.

## What was scanned for

- Deprecated or stale model IDs (Llama-3.3-70B-Instruct-Turbo, Llama 4-era,
  Llama 3.1/3.2-era, DeepSeek-R1, Qwen2.x-era, gemma-4, gpt-oss-20b,
  `intfloat/multilingual-e5-large-instruct`, Mixtral/Mistral-7B-era).
- Legacy v1 Together SDK usage (`together==0.x/1.x` pins, `together.Complete`
  style calls, `HfInference`).
- Hardcoded API keys.
- CI-auth blockers: inline placeholder key assignments and Colab
  `userdata.get` key retrieval, both of which break env-var auth in CI.

## Headline numbers

- **43 of 59 notebooks** carry at least one stale-model or legacy-SDK marker.
- **23 notebooks** still reference `Llama-3.3-70B-Instruct-Turbo`.
- **9 notebooks** embed `intfloat/multilingual-e5-large-instruct`.
- **7 notebooks** reference `DeepSeek-R1`. Live-confirmed broken: the smoke
  test of `Thinking_Augmented_Generation.ipynb` failed with
  `model_not_available` ("Unable to access non-serverless model
  deepseek-ai/DeepSeek-R1"), so all 7 will fail at runtime.
- **6 notebooks** pin the v1 SDK (`together==1.3.3` through `1.5.35`).
- **11 notebooks** cannot authenticate from the environment: 9 assign an
  inline placeholder key (`TOGETHER_API_KEY = "--Your API Key--"` style) and
  2 read keys via Colab `userdata.get`. These need a one-line switch to
  `os.environ` before CI can execute them.
- **No real hardcoded secrets** were found (placeholders only), and no
  `HfInference` or kimi-k2.6-era IDs.

## Worst offenders

- `Evals/Optimizing_LLM_Judges.ipynb`: Llama 4-era, Llama 3.1-era, Qwen2.x,
  Mixtral-era IDs, a `together==1.5.35` pin, and a placeholder key.
- `Agents/Agno/Agents_Agno.ipynb`, `Agents/LangGraph/Agentic_RAG_LangGraph.ipynb`,
  `contextual_rag_on_union/Contextual_RAG_on_Union.ipynb`: three stale-model
  families each.
- The five `Agents/*_Workflow.ipynb` notebooks: stale Llama-3.3 IDs plus
  placeholder keys.

## Per-notebook findings

| Notebook | Tier | Findings (static) |
|---|---|---|
| `Agents/Agno/Agents_Agno.ipynb` | weekly | Llama-3.3-70B-Instruct-Turbo, Llama 3.1/3.2 era, multilingual-e5-large-instruct |
| `Agents/Conditional_Router_Agent_Workflow.ipynb` | weekly | Llama-3.3-70B-Instruct-Turbo, DeepSeek-R1, **placeholder key inline (breaks env auth)** |
| `Agents/DSPy/DSPy_Agents.ipynb` | weekly | Llama-3.3-70B-Instruct-Turbo, Llama 3.1/3.2 era |
| `Agents/DataScienceAgent/Together_Open_DataScience_Agent.ipynb` | excluded | Llama-3.3-70B-Instruct-Turbo, **placeholder key inline (breaks env auth)** |
| `Agents/KlavisAI/Agents_KlavisAI.ipynb` | excluded | Llama-3.3-70B-Instruct-Turbo |
| `Agents/LangGraph/Agentic_RAG_LangGraph.ipynb` | weekly | Llama-3.3-70B-Instruct-Turbo, DeepSeek-R1, multilingual-e5-large-instruct |
| `Agents/LangGraph/LangGraph_Planning_Agent.ipynb` | weekly | Llama-3.3-70B-Instruct-Turbo |
| `Agents/Looping_Agent_Workflow.ipynb` | weekly | Llama-3.3-70B-Instruct-Turbo, **placeholder key inline (breaks env auth)** |
| `Agents/Parallel_Agent_Workflow.ipynb` | weekly | Llama-3.3-70B-Instruct-Turbo, gemma-4, **placeholder key inline (breaks env auth)** |
| `Agents/Parallel_Subtask_Agent_Workflow.ipynb` | weekly | Llama-3.3-70B-Instruct-Turbo, **placeholder key inline (breaks env auth)** |
| `Agents/PydanticAI/PydanticAI_Agents.ipynb` | weekly | Llama-3.3-70B-Instruct-Turbo |
| `Agents/Serial_Chain_Agent_Workflow.ipynb` | weekly | Llama-3.3-70B-Instruct-Turbo, **placeholder key inline (breaks env auth)** |
| `Agents/Together_Open_Deep_Research_CookBook.ipynb` | excluded | Llama-3.3-70B-Instruct-Turbo, **placeholder key inline (breaks env auth)** |
| `Batch_Inference_Evals.ipynb` | excluded | Llama-3.3-70B-Instruct-Turbo, DeepSeek-R1 |
| `Embedding_Visualization.ipynb` | weekly | multilingual-e5-large-instruct |
| `Evals/Classification_Evals.ipynb` | monthly | Llama-3.3-70B-Instruct-Turbo |
| `Evals/Compare_Evals.ipynb` | monthly | Llama-3.3-70B-Instruct-Turbo |
| `Evals/GEPA_Optimization.ipynb` | monthly | Llama-3.3-70B-Instruct-Turbo, gpt-oss-20b, **Colab userdata.get key (breaks CI auth)** |
| `Evals/Optimizing_LLM_Judges.ipynb` | monthly | Llama 4 era, Llama 3.1/3.2 era, Qwen2.x era, Mixtral/Mistral-7B era, **placeholder key inline (breaks env auth)**, `together==1.5.35` pin (v1 SDK era) |
| `Evals/Prompt_Evals.ipynb` | monthly | Qwen2.x era, **placeholder key inline (breaks env auth)** |
| `Finetuning/Continual_Finetuning.ipynb` | excluded | Llama 3.1/3.2 era |
| `Finetuning/DPO_Finetuning.ipynb` | excluded | Llama 3.1/3.2 era |
| `Finetuning/Finetuning_Guide.ipynb` | excluded | Llama 3.1/3.2 era |
| `Finetuning/Function_Calling_Finetuning.ipynb` | excluded | Qwen2.x era |
| `Finetuning/Reasoning_Finetuning.ipynb` | excluded | DeepSeek-R1 |
| `Getting_started_with_Llama4.ipynb` | weekly | Llama 4 era |
| `Knowledge_Graphs_with_Structured_Outputs.ipynb` | weekly | Llama-3.3-70B-Instruct-Turbo, Llama 3.1/3.2 era |
| `LoRA_Finetuning&Inference.ipynb` | excluded | Llama 3.1/3.2 era |
| `LongContext_Finetuning_RepetitionTask.ipynb` | excluded | Llama 3.1/3.2 era, `together==1.3.4` pin (v1 SDK era) |
| `MultiModal_RAG_with_Nvidia_Investor_Slide_Deck.ipynb` | monthly | Qwen2.x era |
| `Multiturn_Conversation_Finetuning.ipynb` | excluded | Llama 3.1/3.2 era, `together==1.3.4` pin (v1 SDK era) |
| `Open_Contextual_RAG.ipynb` | monthly | Llama 4 era, multilingual-e5-large-instruct |
| `PDF_to_Podcast.ipynb` | excluded | Llama-3.3-70B-Instruct-Turbo, Llama 3.1/3.2 era |
| `RAG_with_Reasoning_Models.ipynb` | weekly | DeepSeek-R1, multilingual-e5-large-instruct |
| `Search_with_Reranking.ipynb` | weekly | multilingual-e5-large-instruct |
| `Semantic_Search.ipynb` | weekly | multilingual-e5-large-instruct |
| `Structured_Text_Extraction_from_Images.ipynb` | weekly | Llama-3.3-70B-Instruct-Turbo, Llama 3.1/3.2 era |
| `Summarization_Evaluation.ipynb` | monthly | Llama-3.3-70B-Instruct-Turbo, `together==1.3.3` pin (v1 SDK era) |
| `Summarization_LongContext_Finetuning.ipynb` | excluded | Llama 3.1/3.2 era, `together==1.3.4` pin (v1 SDK era) |
| `Text_RAG.ipynb` | weekly | Llama-3.3-70B-Instruct-Turbo, multilingual-e5-large-instruct |
| `Thinking_Augmented_Generation.ipynb` | weekly | DeepSeek-R1 |
| `contextual_rag_on_union/Contextual_RAG_on_Union.ipynb` | excluded | DeepSeek-R1, multilingual-e5-large-instruct, `together==1.3.10` pin (v1 SDK era) |
| `third_party_integrations/Tool_use_with_Toolhouse.ipynb` | excluded | Llama-3.3-70B-Instruct-Turbo, **Colab userdata.get key (breaks CI auth)** |

Notebooks not listed above had no findings. Tiers come from `.ci/tiers.yaml`
and the per-folder `ci.yaml` manifests. "Era" markers flag model families
that likely have newer replacements; confirm each against the live model
catalog during the funded live triage before editing notebooks.
