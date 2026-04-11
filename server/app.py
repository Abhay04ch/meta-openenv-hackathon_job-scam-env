"""
FastAPI application for the Job Scam Env Environment.

This module creates an HTTP server that exposes the JobScamEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Multi-task support
------------------
A single ``JobScamEnvironment`` class handles all three task variants
(easy / medium / hard) internally via if/elif/else dispatch.  Because
``create_app`` accepts exactly one Action class and one Observation class,
we pass the **unified superset** ``JobScamAction`` and ``JobScamObservation``
which contain the union of all fields across all tasks.

Task variant selection happens at reset time: the client sends
``task_name`` in the reset payload, which the server forwards to
``JobScamEnvironment.reset(task_name=...)``.

Endpoints:
    - POST /reset  : Reset the environment (body may include task_name)
    - POST /step   : Execute an action
    - GET  /state  : Get current environment state
    - GET  /schema : Get action/observation schemas
    - WS   /ws     : WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""
import os
import uvicorn
from dotenv import load_dotenv

try:
    print("importing openenv!!")
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. "
        "Install dependencies with '\n    uv sync\n'"
    ) from e

print("importing job_scam_env!!")
try:
    from job_scam_env.server.models import JobScamAction, JobScamObservation, ActionType
    from job_scam_env.server.job_scam_env_environment import JobScamEnvironment
except (ModuleNotFoundError, ImportError):
    try:
        print("error 1 — trying relative imports")
        from ..models import JobScamAction, JobScamObservation, ActionType
        from .job_scam_env_environment import JobScamEnvironment
    except ImportError:
        print("error 2 — falling back to bare imports")
        from models import JobScamAction, JobScamObservation
        from job_scam_env_environment import JobScamEnvironment

print("SERVER ACTION ENUMS:", [x.value for x in ActionType])

app = create_app(
    JobScamEnvironment,
    JobScamAction,
    JobScamObservation,
    env_name="job_scam_env",
    max_concurrent_envs=1,  # increase to allow more concurrent WebSocket sessions
)


def main(host: str = "0.0.0.0", port: int | None = None):
    """
    Entry point for direct execution via uv run or python -m.

        uv run --project . server
        uv run --project . server --port 8001
        python -m job_scam_env.server.app
    """
    if port is None:
        port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)))
    args = parser.parse_args()
    main(port=args.port) # calls main()