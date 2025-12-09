# Tools and Integrations

Extend your AI applications with code execution, tool calling, and popular agent frameworks.

## 📂 Categories

### 🔧 [Code Execution](Code_Execution/)
Execute Python code safely and scale with Together Code Interpreter
- **Together Code Interpreter**: Safe, scalable code execution for data analysis and visualization
- **OpenEnv Integration**: Code Interpreter as an OpenEnv environment for training pipelines

### 🛠️ [Tool Calling](Tool_Calling/)
Equip LLMs with external tools and capabilities
- **Toolhouse Integration**: Function calling toolkit with pre-built tools (web scraping, image generation, code execution)

### 🤖 [Agent Frameworks](Agent_Frameworks/)
Third-party frameworks that integrate with Together AI

| Framework | Description | Notebooks |
| --------- | ----------- | --------- |
| [LangGraph](Agent_Frameworks/LangGraph/) | Build stateful, multi-actor agents with cycles and persistence | Planning Agent, Agentic RAG |
| [DSPy](Agent_Frameworks/DSPy/) | Programming foundation for LLMs with optimization (MIPRO) | DSPy Agents with 8B→70B optimization |
| [Agno](Agent_Frameworks/Agno/) | Lightweight agent framework | Agent implementation |
| [Composio](Agent_Frameworks/Composio/) | Tool integration platform for agents | Agent with tools |
| [Arcade](Agent_Frameworks/Arcade/) | Unified agent orchestration | Agent workflows |
| [KlavisAI](Agent_Frameworks/KlavisAI/) | AI agent platform | Agent examples |
| [PydanticAI](Agent_Frameworks/PydanticAI/) | Type-safe agents with Pydantic | Pydantic-based agents |

## 🎯 When to Use What

**Code Execution** 
- Data analysis and visualization
- Scientific computing
- File manipulation
- Running arbitrary Python code safely

**Tool Calling**
- Web scraping and API calls
- Image generation from within conversations
- Real-time data fetching
- Extending LLM capabilities with external functions

**Agent Frameworks**
- Complex multi-step workflows
- State management across conversations
- Integration with existing agent ecosystems
- Production-ready agent applications

## 🚀 Getting Started

1. **New to Tools?** Start with [Together Code Interpreter](Code_Execution/Together_Code_Interpreter.ipynb)
2. **Need External APIs?** Check out [Toolhouse](Tool_Calling/Tool_use_with_Toolhouse.ipynb)
3. **Building Complex Agents?** Explore [Agent Frameworks](Agent_Frameworks/)

## Prerequisites

- Together AI API key
- Framework-specific API keys (e.g., Tavily for search, Toolhouse for tools)
- Understanding of agent concepts (recommended: complete [Agents](../Agents/) first)
