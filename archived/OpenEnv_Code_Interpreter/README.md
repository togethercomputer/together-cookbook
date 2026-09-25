# Together Code Interpreter + OpenEnv Integration

## Overview

This demo showcases **Together's Code Interpreter as an OpenEnv environment**, using the same interface as game environments like BlackJack, Chess, and Atari.

## What This Demonstrates

Code Interpreter wrapped in OpenEnv's framework:
- ✅ Same API as all OpenEnv environments (`reset()`, `step()`, `close()`)
- ✅ Server/client architecture (like all OpenEnv environments)
- ✅ Can be used in training pipelines alongside games
- ✅ Modular and swappable

---

## Quick Start

### Installation

```bash
cd OpenEnv_Code_Interpreter
pip install together openenv-core
```

### Running the Demo

```bash
# Terminal 1: Start Code Interpreter OpenEnv server
export TOGETHER_API_KEY="your-api-key"
python -m code_interpreter_env.server.app

# Terminal 2: Run the demo (uses OpenEnv client)
export TOGETHER_API_KEY="your-api-key"
python code_interpreter_demo.py
```

---

## What's Included

- `code_interpreter_env/` - Full OpenEnv environment implementation
  - `server/` - HTTP server wrapping Together's Code Interpreter
  - `client.py` - HTTP client using OpenEnv interface
  - `models.py` - Action/Observation data models
- `code_interpreter_demo.py` - Demo showing the OpenEnv interface

---

## Key Feature: Universal OpenEnv Interface

Code Interpreter uses the **same API** as game environments:

```python
from code_interpreter_env import CodeInterpreterEnv, CodeInterpreterAction

# Same interface as BlackJack, Chess, Atari, etc.
env = CodeInterpreterEnv(base_url="http://localhost:8001")
result = env.reset()              # Start session
result = env.step(action)         # Execute code
env.close()                       # Cleanup
```

This means it can be used anywhere an OpenEnv environment is expected!

---


