"""
PARAGON MAIN - Entry Point and CLI

Commands:
    serve    - Start the API server (with optional GUI)
    replay   - Open a session recording (.rrd file) in Rerun viewer
    import   - Import graph from file (Parquet/CSV/Arrow)
    export   - Export graph to file
    embed    - Backfill embeddings for all nodes
    env      - Show environment detection report
    sessions - List available session recordings

Usage:
    # Start with full GUI (API + React UI + browser) - recommended for development
    python main.py --dev

    # Start with a spec file (will parse and kick off research)
    python main.py --dev --spec path/to/spec.md

    # Start API only
    python main.py serve

    # Replay a session recording (for demos/debugging)
    python main.py replay data/sessions/20251207_200406_f5a1bc3a.rrd

    # List all session recordings
    python main.py sessions

    # Import graph data
    python main.py import nodes.parquet --edges edges.parquet

    # Export graph
    python main.py export --format parquet --output ./data

    # Backfill embeddings
    python main.py embed

    # Show environment
    python main.py env

    # Production server (always includes GUI)
    python main.py serve --prod --workers 4

Spec File Formats:
    - Markdown (.md) - Unstructured text parsed by LLM
    - Plain Text (.txt) - Unstructured text parsed by LLM
    - JSON (.json) - Structured format with explicit fields
"""
import sys
import os
import subprocess
import webbrowser
import time
import signal
from pathlib import Path
from typing import Optional, List

# Add paragon to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Track child processes for cleanup
_child_processes: List[subprocess.Popen] = []


def _cleanup_children(signum=None, frame=None):
    """Clean up child processes on exit."""
    for proc in _child_processes:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    if signum is not None:
        sys.exit(0)


def _start_ui_dev_server() -> Optional[subprocess.Popen]:
    """Start the React UI development server."""
    ui_dir = Path(__file__).parent / "paragon-ui"

    if not ui_dir.exists():
        print("Warning: paragon-ui directory not found")
        return None

    # Check if npm is available
    try:
        subprocess.run(["npm", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: npm not found. Install Node.js to enable the GUI.")
        return None

    # Check if node_modules exists
    if not (ui_dir / "node_modules").exists():
        print("Installing UI dependencies (first-time setup)...")
        subprocess.run(["npm", "install"], cwd=ui_dir, check=True)

    # Start the dev server
    print("Starting React UI...")
    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=ui_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _child_processes.append(proc)
    return proc


def _open_browser(url: str, delay: float = 2.0):
    """Open browser after a short delay to let servers start."""
    time.sleep(delay)
    webbrowser.open(url)


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
    import threading

    # Set up signal handlers for cleanup
    signal.signal(signal.SIGINT, _cleanup_children)
    signal.signal(signal.SIGTERM, _cleanup_children)

    gui_enabled = getattr(args, 'gui', False) or getattr(args, 'prod', False)

    if gui_enabled:
        # Start UI dev server in background
        ui_proc = _start_ui_dev_server()
        if ui_proc:
            # Open browser after servers start
            ui_url = "http://localhost:5173"  # Vite default port
            threading.Thread(
                target=_open_browser,
                args=(ui_url, 3.0),
                daemon=True
            ).start()
            print(f"GUI will open at {ui_url}")

    try:
        if args.prod:
            run_prod_server(args.host, args.port, args.workers)
        else:
            run_dev_server(args.host, args.port)
    finally:
        _cleanup_children()


def cmd_replay(args):
    """Handle replay command - open session recording in Rerun viewer."""
    rrd_path = Path(args.rrd_file)

    if not rrd_path.exists():
        # Check in data/sessions if bare filename given
        sessions_dir = Path(__file__).parent / "data" / "sessions"
        alt_path = sessions_dir / rrd_path.name
        if alt_path.exists():
            rrd_path = alt_path
        else:
            print(f"Error: Recording not found: {args.rrd_file}")
            print(f"Use 'python main.py sessions' to list available recordings")
            sys.exit(1)

    print(f"Opening session recording: {rrd_path}")
    print("(Rerun viewer will open in your browser)")

    try:
        import rerun as rr
        # Open the viewer with the recording
        subprocess.run(["rerun", str(rrd_path)], check=True)
    except ImportError:
        print("Error: rerun-sdk not installed")
        print("Install with: pip install rerun-sdk")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("Error: Failed to open Rerun viewer")
        sys.exit(1)


def cmd_sessions(args):
    """Handle sessions command - list available session recordings."""
    sessions_dir = Path(__file__).parent / "data" / "sessions"

    if not sessions_dir.exists():
        print("No sessions directory found")
        return

    rrd_files = sorted(sessions_dir.glob("*.rrd"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not rrd_files:
        print("No session recordings found")
        return

    print(f"{'='*60}")
    print(f"SESSION RECORDINGS ({len(rrd_files)} total)")
    print(f"{'='*60}")
    print(f"{'Date':<20} {'Session ID':<20} {'Size':>10}")
    print(f"{'-'*20} {'-'*20} {'-'*10}")

    # Show most recent 20
    for rrd_file in rrd_files[:20]:
        name = rrd_file.stem
        parts = name.split("_", 2)
        if len(parts) >= 3:
            date_str = f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:8]} {parts[1][:2]}:{parts[1][2:4]}"
            session_id = parts[2]
        else:
            date_str = "Unknown"
            session_id = name

        size_kb = rrd_file.stat().st_size / 1024
        size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"

        print(f"{date_str:<20} {session_id:<20} {size_str:>10}")

    if len(rrd_files) > 20:
        print(f"\n... and {len(rrd_files) - 20} more")

    print(f"\nTo replay a session:")
    print(f"  python main.py replay {rrd_files[0].name}")


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


def cmd_start_session(args):
    """
    Handle start session command - initialize orchestrator with or without spec.

    This is called when using --dev or --dev --spec <file> to start a session.
    """
    import threading
    import uuid
    from agents.spec_parser import parse_spec_file
    from agents.initial_conversation import (
        create_initial_message_for_spec,
        determine_starting_phase,
    )
    from agents.orchestrator import TDDOrchestrator
    from core.graph_db import ParagonDB
    from agents.tools import set_db

    # Set up signal handlers for cleanup
    signal.signal(signal.SIGINT, _cleanup_children)
    signal.signal(signal.SIGTERM, _cleanup_children)

    # Initialize database
    db = ParagonDB()
    set_db(db)

    session_id = str(uuid.uuid4())
    parsed_spec = None
    starting_phase = None

    # Parse spec file if provided
    if hasattr(args, 'spec') and args.spec:
        spec_path = args.spec
        print(f"Loading spec from: {spec_path}")
        try:
            parsed_spec = parse_spec_file(spec_path)
            print(f"Loaded spec: {parsed_spec.title}")
            print(f"Description: {parsed_spec.description[:100]}...")

            # Determine starting phase
            starting_phase = determine_starting_phase(parsed_spec)
            print(f"Starting phase: {starting_phase}")

            # Show initial message
            initial_message = create_initial_message_for_spec(parsed_spec)
            print(f"\n{initial_message}\n")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error parsing spec: {e}")
            sys.exit(1)

    # Start GUI if requested
    gui_enabled = getattr(args, 'gui', False) or getattr(args, 'dev', False)
    if gui_enabled:
        # Start UI dev server in background
        ui_proc = _start_ui_dev_server()
        if ui_proc:
            # Open browser after servers start
            ui_url = "http://localhost:5173"  # Vite default port
            threading.Thread(
                target=_open_browser,
                args=(ui_url, 3.0),
                daemon=True
            ).start()
            print(f"GUI will open at {ui_url}")

    # Initialize orchestrator
    orchestrator = TDDOrchestrator(
        enable_checkpointing=True,
        persist_to_sqlite=True,
    )

    # If spec was provided, kick off research in background
    if parsed_spec:
        # Build spec string for orchestrator
        spec_content = f"""# {parsed_spec.title}

{parsed_spec.description}

## Requirements
{chr(10).join(f"- {r}" for r in parsed_spec.requirements) if parsed_spec.requirements else "No specific requirements"}

## Must-Have Features
{chr(10).join(f"- {f}" for f in parsed_spec.must_have_features) if parsed_spec.must_have_features else "No specific features"}

## Technical Details
{parsed_spec.technical_details or "No technical details provided"}

## Constraints
{chr(10).join(f"- {c}" for c in parsed_spec.constraints) if parsed_spec.constraints else "No constraints"}
"""
        # TODO: Start orchestrator run in background thread
        # For now, just print that we would start the orchestrator
        print("\n" + "="*60)
        print("ORCHESTRATOR INITIALIZATION")
        print("="*60)
        print(f"Session ID: {session_id}")
        print(f"Spec loaded: Yes")
        print(f"Starting phase: {starting_phase}")
        print("\nOrchestrator would be initialized here with:")
        print(f"  - session_id: {session_id}")
        print(f"  - spec: {len(spec_content)} chars")
        print(f"  - requirements: {len(parsed_spec.requirements)}")
        print("="*60 + "\n")
    else:
        # No spec - show greeting for fresh conversation
        from agents.initial_conversation import get_greeting_message
        greeting = get_greeting_message()
        print("\n" + "="*60)
        print(greeting)
        print("="*60 + "\n")

    # Start the server
    try:
        if args.prod:
            run_prod_server(args.host, args.port, args.workers)
        else:
            run_dev_server(args.host, args.port)
    finally:
        _cleanup_children()


def main():
    """Main entry point with subcommands."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Paragon - Graph-Native TDD Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Top-level --dev flag for quick startup with full GUI
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Start in dev mode: API + React UI + open browser"
    )

    # Top-level --spec flag for providing a spec file
    parser.add_argument(
        "--spec",
        type=str,
        help="Path to spec file (markdown, text, or JSON) to initialize the session"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    serve_parser.add_argument("--workers", type=int, default=4, help="Number of workers (prod)")
    serve_parser.add_argument("--prod", action="store_true", help="Run in production mode")
    serve_parser.add_argument("--gui", action="store_true", help="Start React UI and open browser")
    serve_parser.set_defaults(func=cmd_serve)

    # replay command
    replay_parser = subparsers.add_parser("replay", help="Replay a session recording")
    replay_parser.add_argument("rrd_file", help="Path to .rrd file (or filename in data/sessions)")
    replay_parser.set_defaults(func=cmd_replay)

    # sessions command
    sessions_parser = subparsers.add_parser("sessions", help="List available session recordings")
    sessions_parser.set_defaults(func=cmd_sessions)

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

    # Handle --dev flag (top-level shortcut)
    if args.dev:
        # Use cmd_start_session for --dev mode (supports --spec)
        args.command = "start_session"
        args.host = "127.0.0.1"
        args.port = 8000
        args.workers = 1
        args.prod = False
        args.gui = True
        args.func = cmd_start_session
    elif args.command is None:
        # Default to serve (API only)
        args.command = "serve"
        args.host = "127.0.0.1"
        args.port = 8000
        args.workers = 1
        args.prod = False
        args.gui = False
        args.func = cmd_serve

    args.func(args)


if __name__ == "__main__":
    main()
