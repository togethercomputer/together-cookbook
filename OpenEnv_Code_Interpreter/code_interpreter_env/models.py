# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Code Interpreter Environment.

The Code Interpreter environment wraps Together AI's Code Interpreter API,
allowing agents to execute Python code in a secure, sandboxed environment.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from core.env_server.types import Action, Observation
except ImportError:
    from openenv_core.env_server.types import Action, Observation


@dataclass(kw_only=True)
class CodeInterpreterAction(Action):
    """
    Action for the Code Interpreter environment.
    
    Attributes:
        code: Python code to execute in the interpreter
        install_packages: Optional list of packages to install before execution
    """
    code: str
    install_packages: Optional[list[str]] = None


@dataclass(kw_only=True)
class CodeInterpreterObservation(Observation):
    """
    Observation from the Code Interpreter environment.
    
    Attributes:
        output: Standard output from code execution
        error: Error message if execution failed
        execution_time: Time taken to execute (in seconds)
        session_id: Current Code Interpreter session ID
        metadata: Additional execution metadata (files created, etc.)
    """
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

