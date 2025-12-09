# Agents

Build autonomous AI workflows using fundamental agent patterns with Together AI. Learn core workflow types and see them applied in specialized agent implementations.

## 📚 Core Agent Workflows

| Workflow | Description | Open |
| -------- | ----------- | ---- |
| [Serial Chain Agent](Serial_Chain_Agent_Workflow.ipynb) | Chain multiple LLM calls sequentially to process complex tasks | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Serial_Chain_Agent_Workflow.ipynb) |
| [Conditional Router Agent](Conditional_Router_Agent_Workflow.ipynb) | Create an agent that routes tasks to specialized models | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Conditional_Router_Agent_Workflow.ipynb) |
| [Parallel Agent Workflow](Parallel_Agent_Workflow.ipynb) | Run multiple LLMs in parallel and aggregate their solutions | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Parallel_Agent_Workflow.ipynb) |
| [Orchestrator Subtask Agent](Parallel_Subtask_Agent_Workflow.ipynb) | Break down tasks into parallel subtasks to be executed by LLMs | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Parallel_Subtask_Agent_Workflow.ipynb) |
| [Looping Agent Workflow](Looping_Agent_Workflow.ipynb) | Build an agent that iteratively improves responses | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Looping_Agent_Workflow.ipynb) |

## 🎯 Specialized Agent Implementations

| Agent | Description | Open |
| ----- | ----------- | ---- |
| [Together Open Deep Research](Together_Open_Deep_Research_CookBook.ipynb) | Open-source deep research implementation with multi-step web search | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Together_Open_Deep_Research_CookBook.ipynb) |
| [Data Science Agent](DataScienceAgent/Together_Open_DataScience_Agent.ipynb) | ReAct agent for autonomous data analysis using Together Code Interpreter | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/DataScienceAgent/Together_Open_DataScience_Agent.ipynb) |

## 🔧 Workflow Patterns Explained

### Serial Chain
- **Use Case**: Multi-step transformations (e.g., PDF → Summary → Outline → Script)
- **Pattern**: Output₁ → Input₂ → Output₂ → Input₃ → Final Output
- **Example**: Converting PDFs to podcasts with intermediate processing steps

### Conditional Router
- **Use Case**: Task-specific model selection (e.g., math → math model, code → code model)
- **Pattern**: Classifier → Route → Specialized Model → Result
- **Example**: Customer support routing queries to appropriate expert models

### Parallel Execution
- **Use Case**: Ensemble methods, diverse perspectives (e.g., multiple reviewers)
- **Pattern**: Input → [Model₁, Model₂, Model₃] → Aggregate → Final Output
- **Example**: Mixture of Agents for improved answer quality

### Orchestrator Subtask
- **Use Case**: Complex tasks decomposable into independent subtasks
- **Pattern**: Planner → [Subtask₁, Subtask₂, Subtask₃] → Synthesizer → Result
- **Example**: Research questions broken into parallel investigations

### Looping
- **Use Case**: Iterative refinement, self-critique
- **Pattern**: Generate → Evaluate → Improve → Repeat until satisfied
- **Example**: Code generation with error correction

## 🎓 Learning Path

1. **Start Here**: [Serial Chain](Serial_Chain_Agent_Workflow.ipynb) - Simplest pattern
2. **Add Logic**: [Conditional Router](Conditional_Router_Agent_Workflow.ipynb) - Decision making
3. **Scale Up**: [Parallel](Parallel_Agent_Workflow.ipynb) - Concurrent execution
4. **Get Advanced**: [Orchestrator](Parallel_Subtask_Agent_Workflow.ipynb) + [Looping](Looping_Agent_Workflow.ipynb)
5. **Apply**: [Deep Research](Together_Open_Deep_Research_CookBook.ipynb) or [Data Science](DataScienceAgent/Together_Open_DataScience_Agent.ipynb)

## 🛠️ Agent Frameworks

Looking for pre-built agent frameworks? Check out [Tools and Integrations](../Tools_and_Integrations/Agent_Frameworks/) for:
- LangGraph - Stateful, multi-actor agents
- DSPy - Optimized agent prompts
- Agno, Composio, Arcade, KlavisAI, PydanticAI - Various frameworks

## 🔑 Key Technologies

- **Together Code Interpreter**: Safe code execution for data analysis agents
- **Reasoning Models**: DeepSeek R1, Llama 3.3 70B for complex reasoning
- **Structured Outputs**: JSON schemas for reliable agent communications
- **Function Calling**: Tool use for extended capabilities

## Prerequisites

- Together AI API key
- Understanding of LLM prompting
- Python basics
- For specialized agents: domain knowledge (research, data science)
