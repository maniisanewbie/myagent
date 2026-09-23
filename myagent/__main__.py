"""Run the agent server: `python -m myagent`."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "myagent.server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
