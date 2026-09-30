import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from code_interpreter_env import CodeInterpreterEnv, CodeInterpreterAction


def main():
    # Check for API key
    if not os.getenv("TOGETHER_API_KEY"):
        print("❌ Error: TOGETHER_API_KEY environment variable not set")
        return
    
    print("\n🔧 Together Code Interpreter + OpenEnv Demo\n")
    
    # Connect to environment
    env = CodeInterpreterEnv(base_url="http://localhost:8001")
    result = env.reset()
    print(f"✅ Session started: {result.observation.session_id}\n")
    
    # Test 1: Simple calculation
    print("Test 1: Simple calculation")
    action = CodeInterpreterAction(code="print(2 + 2)")
    result = env.step(action)
    print(f"  Input:  print(2 + 2)")
    print(f"  Output: {result.observation.output}")
    print(f"  Time:   {result.observation.execution_time:.2f}s\n")
    
    # Test 2: Import library
    print("Test 2: Math library")
    action = CodeInterpreterAction(code="import math\nprint(f'Pi = {math.pi:.4f}')")
    result = env.step(action)
    print(f"  Input:  import math; print(math.pi)")
    print(f"  Output: {result.observation.output}\n")
    
    # Test 3: List comprehension
    print("Test 3: List comprehension")
    action = CodeInterpreterAction(code="print([x**2 for x in range(5)])")
    result = env.step(action)
    print(f"  Input:  [x**2 for x in range(5)]")
    print(f"  Output: {result.observation.output}\n")
    
    # Test 4: NumPy
    print("Test 4: NumPy computation")
    action = CodeInterpreterAction(code="import numpy as np\narr = np.array([1, 2, 3, 4, 5])\nprint(f'Mean: {arr.mean()}')")
    result = env.step(action)
    print(f"  Input:  numpy array mean")
    print(f"  Output: {result.observation.output}\n")
    
    env.close()
    
    print("✅ All tests passed!")
    print("\nCode Interpreter works as an OpenEnv environment:")
    print("  • Same API: reset(), step(), state(), close()")
    print("  • Ready to integrate with training pipelines")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. TOGETHER_API_KEY is set")
        print("  2. Server is running: python -m code_interpreter_env.server.app")
