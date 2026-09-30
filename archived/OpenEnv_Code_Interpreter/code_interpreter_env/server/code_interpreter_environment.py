# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Together Code Interpreter Environment Implementation.

This environment wraps Together AI's Code Interpreter API, allowing agents
to execute Python code in a secure, sandboxed environment.
"""

import os
import time
from uuid import uuid4
from typing import Optional

try:
    from core.env_server.interfaces import Environment
    from core.env_server.types import State
    from ..models import CodeInterpreterAction, CodeInterpreterObservation
except ImportError:
    from openenv_core.env_server.interfaces import Environment
    from openenv_core.env_server.types import State
    from ..models import CodeInterpreterAction, CodeInterpreterObservation

try:
    from together import Together
except ImportError:
    raise ImportError(
        "Together SDK is required for Code Interpreter environment. "
        "Install with: uv pip install together"
    )


class CodeInterpreterEnvironment(Environment):
    """
    Code Interpreter environment using Together AI's API.

    This environment executes Python code via Together's Code Interpreter API.
    Useful for:
    - Reward function computation
    - Data processing and analysis
    - Environment logic that requires computation
    - Real-time metrics and validation

    Example:
        >>> env = CodeInterpreterEnvironment(api_key="your-api-key")
        >>> obs = env.reset()
        >>> action = CodeInterpreterAction(code="print(2 + 2)")
        >>> obs = env.step(action)
        >>> print(obs.output)  # "4"
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Code Interpreter environment.

        Args:
            api_key: Together API key (defaults to TOGETHER_API_KEY env var)
        
        Raises:
            ValueError: If no API key is provided
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._session_id: Optional[str] = None
        self._api_key = api_key or os.getenv("TOGETHER_API_KEY")
        
        if not self._api_key:
            raise ValueError(
                "TOGETHER_API_KEY is required. Set it as an environment variable or pass it to the constructor."
            )
        
        self._client = Together(api_key=self._api_key)

    def reset(self) -> CodeInterpreterObservation:
        """
        Reset the environment and create a new Code Interpreter session.

        Returns:
            CodeInterpreterObservation with session ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        session_id = f"session-{uuid4().hex[:8]}"
        self._session_id = session_id
        
        return CodeInterpreterObservation(
            output=f"✅ Code Interpreter session '{session_id}' ready!",
            session_id=session_id,
            done=False,
            reward=0.0,
        )

    def step(self, action: CodeInterpreterAction) -> CodeInterpreterObservation:  # type: ignore[override]
        """
        Execute Python code in the Code Interpreter session.

        Args:
            action: CodeInterpreterAction containing code to execute

        Returns:
            CodeInterpreterObservation with execution results
        """
        self._state.step_count += 1
        start_time = time.time()
        
        try:
            # Execute the code using Together's Code Interpreter API
            response = self._client.code_interpreter.run(
                code=action.code,
                language="python"
            )
            
            execution_time = time.time() - start_time
            
            # Extract output from response
            output = self._extract_output(response)
            error_msg = self._extract_error(response)
            
            return CodeInterpreterObservation(
                output=output,
                error=error_msg,
                execution_time=execution_time,
                session_id=response.data.session_id if hasattr(response, 'data') else self._session_id,
                metadata={
                    "status": response.data.status if hasattr(response, 'data') else "completed",
                    "step": self._state.step_count
                },
                done=False,
                reward=1.0 if error_msg is None else 0.0,
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return CodeInterpreterObservation(
                output="",
                error=str(e),
                execution_time=execution_time,
                session_id=self._session_id,
                metadata={"step": self._state.step_count},
                done=False,
                reward=0.0,
            )

    def _extract_output(self, response) -> str:
        """Extract output from Together API response."""
        try:
            if hasattr(response, 'data') and hasattr(response.data, 'outputs'):
                outputs = response.data.outputs
                # Concatenate all stdout outputs
                result = []
                for output in outputs:
                    if hasattr(output, 'type') and output.type == 'stdout':
                        result.append(output.data)
                return ''.join(result).strip()
            return "Execution completed (no output)"
        except Exception:
            return "Execution completed (no output captured)"

    def _extract_error(self, response) -> Optional[str]:
        """Extract errors from Together API response."""
        try:
            if hasattr(response, 'data') and hasattr(response.data, 'errors') and response.data.errors:
                return str(response.data.errors)
            return None
        except Exception:
            return None

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

