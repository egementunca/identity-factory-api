#!/usr/bin/env python3
"""
Startup script for the Identity Circuit Factory API server.

This script provides a convenient way to start the API server with
common configuration options and environment setup.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from identity_factory.api.server import run_server


def main():
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(
        description="Identity Circuit Factory API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with default settings
  python start_api.py

  # Start server on specific host and port
  python start_api.py --host 0.0.0.0 --port 8080

  # Start server with auto-reload for development
  python start_api.py --reload --debug

  # Start server with multiple workers
  python start_api.py --workers 4

  # Start server with custom database path
  python start_api.py --db-path /path/to/custom.db
        """,
    )

    # Server configuration
    parser.add_argument(
        "--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )

    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes (default: 1)"
    )

    # Factory configuration
    parser.add_argument(
        "--db-path", help="Database file path (default: identity_circuits.db)"
    )

    parser.add_argument(
        "--max-inverse-gates",
        type=int,
        default=40,
        help="Maximum inverse gates for synthesis (default: 40)",
    )

    parser.add_argument(
        "--max-equivalents",
        type=int,
        default=10000,
        help="Maximum equivalent circuits per group (default: 10000)",
    )

    parser.add_argument(
        "--sat-solver",
        default="minisat-gh",
        help="SAT solver to use (default: minisat-gh)",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--no-post-processing",
        action="store_true",
        help="Disable post-processing (simplification)",
    )

    parser.add_argument(
        "--no-unrolling",
        action="store_true",
        help="Disable unrolling (equivalent generation)",
    )

    args = parser.parse_args()

    # Set environment variables for factory configuration
    if args.db_path:
        os.environ["IDENTITY_FACTORY_DB_PATH"] = args.db_path

    os.environ["IDENTITY_FACTORY_MAX_INVERSE_GATES"] = str(args.max_inverse_gates)
    os.environ["IDENTITY_FACTORY_MAX_EQUIVALENTS"] = str(args.max_equivalents)
    os.environ["IDENTITY_FACTORY_SAT_SOLVER"] = args.sat_solver
    os.environ["IDENTITY_FACTORY_LOG_LEVEL"] = args.log_level

    if args.no_post_processing:
        os.environ["IDENTITY_FACTORY_ENABLE_POST_PROCESSING"] = "false"

    if args.no_unrolling:
        os.environ["IDENTITY_FACTORY_ENABLE_UNROLLING"] = "false"

    # Print startup information
    print("🔬 Identity Circuit Factory API Server")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print(f"Auto-reload: {args.reload}")
    print(f"Workers: {args.workers}")
    print(f"Database: {args.db_path or 'identity_circuits.db'}")
    print(f"SAT Solver: {args.sat_solver}")
    print(f"Log Level: {args.log_level}")
    print(f"Post-processing: {not args.no_post_processing}")
    print(f"Unrolling: {not args.no_unrolling}")
    print("=" * 50)

    # Start the server
    try:
        run_server(
            host=args.host,
            port=args.port,
            reload=args.reload,
            debug=args.debug,
            workers=args.workers,
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
