"""Starts the API server: python -m app [--host 127.0.0.1] [--port 8000]

Same as `uvicorn app.main:app`, but it also runs under PyCharm's debugger. uvicorn's Server.run() calls
asyncio.run(..., loop_factory=...), and the debugger replaces asyncio.run with a version that doesn't accept
loop_factory. Calling asyncio.run with just the coroutine works with both.
"""

import argparse
import asyncio

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app", description="Run the Schematic Symbol Trainer API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = uvicorn.Server(uvicorn.Config("app.main:app", host=args.host, port=args.port))
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
