"""
PARAGON MAIN - Entry Point

Start the Paragon API server using Granian.

Usage:
    # Development (auto-reload)
    python main.py

    # Production (via granian directly)
    granian --interface asgi api.routes:app

    # With specific workers
    granian --interface asgi --workers 4 api.routes:app

Granian Benefits:
- Rust-based HTTP server (fast!)
- Native ASGI support
- Built-in HTTP/2
- Low memory footprint
"""
import sys
import os
from pathlib import Path

# Add paragon to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def run_dev_server(host: str = "127.0.0.1", port: int = 8000):
    """Run development server with Granian."""
    from granian import Granian
    from granian.constants import Interfaces

    print(f"Starting Paragon API server on {host}:{port}")
    print("Press Ctrl+C to stop")

    granian = Granian(
        target="api.routes:app",
        address=host,
        port=port,
        interface=Interfaces.ASGI,
        workers=1,
        reload=True,  # Enable auto-reload for development
    )

    granian.serve()


def run_prod_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 4):
    """Run production server with Granian."""
    from granian import Granian
    from granian.constants import Interfaces

    print(f"Starting Paragon API server on {host}:{port} with {workers} workers")

    granian = Granian(
        target="api.routes:app",
        address=host,
        port=port,
        interface=Interfaces.ASGI,
        workers=workers,
        reload=False,
    )

    granian.serve()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Paragon API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--prod", action="store_true", help="Run in production mode")

    args = parser.parse_args()

    if args.prod:
        run_prod_server(args.host, args.port, args.workers)
    else:
        run_dev_server(args.host, args.port)


if __name__ == "__main__":
    main()
