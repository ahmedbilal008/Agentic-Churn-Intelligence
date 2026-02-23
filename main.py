"""
Churn Intelligence Platform — Entry Point
==========================================

Quick launcher for common operations.
For production use, run individual components:
  - Pipeline:  uv run python -m src.pipelines.training_pipeline --stage full
  - MCP:       uv run python -m src.mcp.server --transport sse
  - DVC:       uv run dvc repro
"""

import sys


def main() -> None:
    """Run a component based on command-line argument."""
    if len(sys.argv) < 2:
        print(
            "Churn Intelligence Platform\n"
            "Usage:\n"
            "  uv run python main.py train    — Run full training pipeline\n"
            "  uv run python main.py serve    — Start MCP server (SSE)\n"
            "  uv run python main.py evaluate — Run evaluation stage\n"
        )
        return

    command = sys.argv[1]

    if command == "train":
        from src.services.pipelines.training_pipeline import run_full_pipeline

        results = run_full_pipeline()
        print(f"\nBest model: {results['training']['best_model']}")
        print(f"Metrics: {results['training']['metrics']}")

    elif command == "serve":
        import os
        from src.interfaces.mcp.server import mcp

        transport = sys.argv[2] if len(sys.argv) > 2 else "sse"
        # Heroku sets PORT dynamically; MCP_PORT is the local override; fallback 8000
        port = int(os.environ.get("PORT") or os.environ.get("MCP_PORT") or 8000)
        host = os.environ.get("MCP_HOST", "0.0.0.0")
        print(f"Starting MCP server with {transport} transport on {host}:{port}...")
        if transport == "stdio":
            mcp.run(transport="stdio")
        else:
            mcp.run(transport="sse", host=host, port=port)

    elif command == "evaluate":
        from src.services.pipelines.training_pipeline import stage_evaluate

        results = stage_evaluate()
        print(f"\nEvaluation: {results}")

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
