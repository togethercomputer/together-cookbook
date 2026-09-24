# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Code Interpreter Environment.

This module creates an HTTP server that exposes the CodeInterpreterEnvironment
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8001

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8001 --workers 4

    # Or run directly:
    python -m envs.code_interpreter_env.server.app
"""

try:
    from core.env_server.http_server import create_app
    from ..models import CodeInterpreterAction, CodeInterpreterObservation
    from .code_interpreter_environment import CodeInterpreterEnvironment
except ImportError:
    from openenv_core.env_server.http_server import create_app
    from ..models import CodeInterpreterAction, CodeInterpreterObservation
    from .code_interpreter_environment import CodeInterpreterEnvironment

# Create the environment instance
env = CodeInterpreterEnvironment()

# Create the app with web interface and README integration
app = create_app(env, CodeInterpreterAction, CodeInterpreterObservation, env_name="code_interpreter_env")


def main():
    """
    Entry point for direct execution.

    This function enables running the server without Docker:
        python -m envs.code_interpreter_env.server.app
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()

