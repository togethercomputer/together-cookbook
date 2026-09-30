# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Together Code Interpreter HTTP Client.

This module provides the client for connecting to a Code Interpreter Environment server
over HTTP.
"""

from typing import Any, Dict

try:
    from core.client_types import StepResult
    from core.env_server.types import State
    from core.http_env_client import HTTPEnvClient
    from .models import CodeInterpreterAction, CodeInterpreterObservation
except ImportError:
    from openenv_core.client_types import StepResult
    from openenv_core.env_server.types import State
    from openenv_core.http_env_client import HTTPEnvClient
    from .models import CodeInterpreterAction, CodeInterpreterObservation


class CodeInterpreterEnv(HTTPEnvClient[CodeInterpreterAction, CodeInterpreterObservation]):
    """
    HTTP client for the Code Interpreter Environment.

    This client connects to a CodeInterpreterEnvironment HTTP server and provides
    methods to interact with Together AI's Code Interpreter API through OpenEnv.

    Example:
        >>> # Connect to a running server
        >>> client = CodeInterpreterEnv(base_url="http://localhost:8001")
        >>> result = client.reset()
        >>> print(result.observation.session_id)
        >>>
        >>> # Execute Python code
        >>> action = CodeInterpreterAction(code="print('Hello from Code Interpreter!')")
        >>> result = client.step(action)
        >>> print(result.observation.output)
        "Hello from Code Interpreter!"

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CodeInterpreterEnv.from_docker_image("code-interpreter-env:latest")
        >>> result = client.reset()
        >>> action = CodeInterpreterAction(code="import numpy as np\\nprint(np.__version__)")
        >>> result = client.step(action)
    """

    def _step_payload(self, action: CodeInterpreterAction) -> Dict:
        """
        Convert CodeInterpreterAction to JSON payload for step request.

        Args:
            action: CodeInterpreterAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload = {
            "code": action.code,
        }
        if action.install_packages:
            payload["install_packages"] = action.install_packages
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[CodeInterpreterObservation]:
        """
        Parse server response into StepResult[CodeInterpreterObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with CodeInterpreterObservation
        """
        obs_data = payload.get("observation", {})
        observation = CodeInterpreterObservation(
            output=obs_data.get("output", ""),
            error=obs_data.get("error"),
            execution_time=obs_data.get("execution_time", 0.0),
            session_id=obs_data.get("session_id"),
            metadata=obs_data.get("metadata", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

