"""
PARAGON MAIN - Entry Point and CLI

Commands:
    serve    - Start the API server (default)
    import   - Import graph from file (Parquet/CSV/Arrow)
    export   - Export graph to file
    embed    - Backfill embeddings for all nodes
    env      - Show environment detection report

Usage:
    # Start development server
    python main.py serve

    # Import graph data
    python main.py import nodes.parquet --edges edges.parquet

    # Export graph
    python main.py export --format parquet --output ./data

    # Backfill embeddings
    python main.py embed

    # Show environment
    python main.py env

    # Production server
    python main.py serve --prod --workers 4
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
        reload=True,
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


def cmd_serve(args):
    """Handle serve command."""
    if args.prod:
        run_prod_server(args.host, args.port, args.workers)
    else:
        run_dev_server(args.host, args.port)


def cmd_import(args):
    """Handle import command - bulk import graph from files."""
    from core.graph_db import ParagonDB
    from agents.tools import set_db, import_graph_from_file

    print(f"Importing graph from {args.nodes_file}...")

    # Initialize database
    db = ParagonDB()
    set_db(db)

    # Determine format
    file_format = args.format
    if not file_format:
        suffix = Path(args.nodes_file).suffix.lower()
        file_format = {
            '.parquet': 'parquet',
            '.csv': 'csv',
            '.arrow': 'arrow',
        }.get(suffix, 'parquet')

    result = import_graph_from_file(
        nodes_path=args.nodes_file,
        edges_path=args.edges_file,
        format=file_format,
    )

    if result.success:
        print(f"Imported {result.nodes_imported} nodes, {result.edges_imported} edges")
    else:
        print(f"Import failed: {result.message}")
        sys.exit(1)


def cmd_export(args):
    """Handle export command - export graph to files."""
    from core.graph_db import ParagonDB
    from agents.tools import set_db, get_db, export_graph_to_parquet

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = str(output_dir / "nodes.parquet")
    edges_path = str(output_dir / "edges.parquet")

    print(f"Exporting graph to {output_dir}...")

    result = export_graph_to_parquet(nodes_path, edges_path)

    if result.get("success"):
        print(f"Exported {result['nodes_exported']} nodes, {result['edges_exported']} edges")
        print(f"  Nodes: {nodes_path}")
        print(f"  Edges: {edges_path}")
    else:
        print(f"Export failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


def cmd_embed(args):
    """Handle embed command - backfill embeddings for all nodes."""
    from core.graph_db import ParagonDB
    from agents.tools import set_db, get_db

    print("Backfilling embeddings for all nodes...")

    # Check if embeddings available
    try:
        from core.embeddings import is_available
        if not is_available():
            print("Error: sentence-transformers not installed")
            print("Install with: pip install sentence-transformers")
            sys.exit(1)
    except ImportError:
        print("Error: core.embeddings module not found")
        sys.exit(1)

    db = get_db()
    count = db.update_all_embeddings(batch_size=args.batch_size)

    print(f"Computed {count} embeddings")
    if count == 0:
        print("(All nodes already have embeddings or no content to embed)")


def cmd_env(args):
    """Handle env command - show environment detection report."""
    try:
        from infrastructure.environment import EnvironmentDetector
    except ImportError:
        print("Error: infrastructure.environment module not found")
        sys.exit(1)

    detector = EnvironmentDetector()
    report = detector.detect()

    print("=" * 50)
    print("PARAGON ENVIRONMENT REPORT")
    print("=" * 50)
    print(f"OS:              {report.os_name}")
    print(f"Python:          {report.python_version}")
    print(f"RAM:             {report.ram_gb:.1f} GB")
    print(f"Disk Free:       {report.disk_free_gb:.1f} GB")
    print(f"GPU Available:   {'Yes' if report.gpu_available else 'No'}")
    if report.gpu_available and report.gpu_name:
        print(f"GPU Name:        {report.gpu_name}")
    print(f"Network:         {'Available' if report.network_available else 'Unavailable'}")
    print(f"Git Repo:        {'Yes' if report.git_repo_present else 'No'}")
    print(f"Working Dir:     {report.working_directory}")
    print("=" * 50)

    # Show module availability
    print("\nModule Availability:")
    modules = [
        ("sentence-transformers", "core.embeddings"),
        ("rerun-sdk", "infrastructure.rerun_logger"),
        ("langgraph", "langgraph.graph"),
        ("rustworkx", "rustworkx"),
        ("polars", "polars"),
        ("msgspec", "msgspec"),
    ]
    for name, module in modules:
        try:
            __import__(module.split(".")[0])
            print(f"  [{'+'}] {name}")
        except ImportError:
            print(f"  [{'x'}] {name} (not installed)")


def main():
    """Main entry point with subcommands."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Paragon - Graph-Native TDD Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    serve_parser.add_argument("--workers", type=int, default=4, help="Number of workers (prod)")
    serve_parser.add_argument("--prod", action="store_true", help="Run in production mode")
    serve_parser.set_defaults(func=cmd_serve)

    # import command
    import_parser = subparsers.add_parser("import", help="Import graph from file")
    import_parser.add_argument("nodes_file", help="Path to nodes file")
    import_parser.add_argument("--edges", dest="edges_file", help="Path to edges file (optional)")
    import_parser.add_argument("--format", choices=["parquet", "csv", "arrow"], help="File format")
    import_parser.set_defaults(func=cmd_import)

    # export command
    export_parser = subparsers.add_parser("export", help="Export graph to files")
    export_parser.add_argument("--output", "-o", default="./export", help="Output directory")
    export_parser.add_argument("--format", choices=["parquet", "csv", "arrow"], default="parquet")
    export_parser.set_defaults(func=cmd_export)

    # embed command
    embed_parser = subparsers.add_parser("embed", help="Backfill embeddings for nodes")
    embed_parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    embed_parser.set_defaults(func=cmd_embed)

    # env command
    env_parser = subparsers.add_parser("env", help="Show environment report")
    env_parser.set_defaults(func=cmd_env)

    args = parser.parse_args()

    if args.command is None:
        # Default to serve
        args.command = "serve"
        args.host = "127.0.0.1"
        args.port = 8000
        args.workers = 1
        args.prod = False
        args.func = cmd_serve

    args.func(args)


if __name__ == "__main__":
    main()
