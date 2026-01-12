"""
Command Line Interface for Identity Circuit Factory.
Provides commands for generating, managing, and analyzing identity circuits.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .database import CircuitDatabase
from .factory_manager import FactoryConfig, IdentityFactory
from .seed_generator import SeedGenerator
from .unroller import CircuitUnroller

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Identity Circuit Factory - Generate and manage identity circuits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--db-path",
        default="identity_circuits.db",
        help="Path to the database file (default: identity_circuits.db)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate identity circuits")
    gen_parser.add_argument("width", type=int, help="Number of qubits")
    gen_parser.add_argument(
        "gate_count", type=int, help="Number of gates in forward circuit"
    )
    gen_parser.add_argument(
        "--max-inverse-gates",
        type=int,
        default=40,
        help="Maximum gates in inverse circuit (default: 40)",
    )
    gen_parser.add_argument(
        "--no-unroll", action="store_true", help="Skip unrolling step"
    )
    gen_parser.add_argument(
        "--no-post-process", action="store_true", help="Skip post-processing step"
    )
    gen_parser.add_argument(
        "--solver", default="minisat-gh", help="SAT solver to use (default: minisat-gh)"
    )
    gen_parser.add_argument("--output", "-o", help="Output file for results")

    # List command
    list_parser = subparsers.add_parser(
        "list", help="List dimension groups and circuits"
    )
    list_parser.add_argument("--width", type=int, help="Filter by width")
    list_parser.add_argument("--gate-count", type=int, help="Filter by gate count")
    list_parser.add_argument(
        "--show-representatives",
        action="store_true",
        help="Show representative circuits",
    )
    list_parser.add_argument(
        "--show-equivalents", action="store_true", help="Show equivalent circuits count"
    )
    list_parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )

    # Unroll command
    unroll_parser = subparsers.add_parser("unroll", help="Unroll dimension groups")
    unroll_parser.add_argument(
        "dim_group_id",
        type=int,
        nargs="?",
        help="Dimension group ID to unroll (optional)",
    )
    unroll_parser.add_argument(
        "--all", action="store_true", help="Unroll all unprocessed dimension groups"
    )
    unroll_parser.add_argument(
        "--types",
        nargs="+",
        choices=[
            "swap",
            "rotation",
            "permutation",
            "reverse",
            "local_unroll",
            "full_unroll",
        ],
        help="Unroll types to apply",
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show factory statistics")
    stats_parser.add_argument(
        "--detailed", action="store_true", help="Show detailed statistics"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a dimension group")
    analyze_parser.add_argument("dim_group_id", type=int, help="Dimension group ID")
    analyze_parser.add_argument("--export", help="Export analysis to file")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch generate circuits")
    batch_parser.add_argument("config_file", help="JSON config file with dimensions")
    batch_parser.add_argument(
        "--max-inverse-gates",
        type=int,
        default=40,
        help="Maximum gates in inverse circuit",
    )
    batch_parser.add_argument("--output-dir", help="Output directory for results")

    return parser


def format_dimension_groups(
    dim_groups: List,
    show_representatives: bool = False,
    show_equivalents: bool = False,
    format_type: str = "table",
) -> str:
    """Format dimension groups for display."""
    if format_type == "json":
        return json.dumps(
            [
                {
                    "id": dg.id,
                    "width": dg.width,
                    "gate_count": dg.gate_count,
                    "circuit_count": dg.circuit_count,
                    "is_processed": dg.is_processed,
                }
                for dg in dim_groups
            ],
            indent=2,
        )

    # Table format
    lines = []
    lines.append("Dimension Groups:")
    lines.append("=" * 80)
    lines.append(
        f"{'ID':<5} {'Width':<6} {'Gates':<6} {'Circuits':<10} {'Processed':<10}"
    )
    lines.append("-" * 80)

    for dg in dim_groups:
        lines.append(
            f"{dg.id:<5} {dg.width:<6} {dg.gate_count:<6} {dg.circuit_count:<10} {'Yes' if dg.is_processed else 'No':<10}"
        )

    return "\n".join(lines)


def cmd_generate(args, factory: IdentityFactory) -> int:
    """Handle generate command."""
    try:
        logger.info(f"Generating identity circuit ({args.width}, {args.gate_count})")

        result = factory.generate_identity_circuit(
            width=args.width,
            gate_count=args.gate_count,
            enable_unrolling=not args.no_unroll,
            enable_post_processing=not args.no_post_process,
            max_inverse_gates=args.max_inverse_gates,
            solver=args.solver,
        )

        if result["success"]:
            print(
                f"✓ Successfully generated identity circuit ({args.width}, {args.gate_count})"
            )
            print(f"  Generation time: {result['total_time']:.2f}s")

            if "seed_generation" in result:
                seed_result = result["seed_generation"]
                print(f"  Circuit ID: {seed_result.circuit_id}")
                print(f"  Dimension Group ID: {seed_result.dim_group_id}")
                if (
                    hasattr(seed_result, "representative_id")
                    and seed_result.representative_id
                ):
                    print(f"  Representative ID: {seed_result.representative_id}")

            if "unrolling" in result and isinstance(result["unrolling"], dict):
                unroll_result = result["unrolling"]
                if "total_equivalents" in unroll_result:
                    print(f"  Total equivalents: {unroll_result['total_equivalents']}")

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"  Results saved to: {args.output}")
        else:
            print(f"✗ Generation failed: {result.get('error', 'Unknown error')}")
            return 1

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        print(f"✗ Generation failed: {e}")
        return 1

    return 0


def cmd_list(args, factory: IdentityFactory) -> int:
    """Handle list command."""
    try:
        db = factory.db

        # Get dimension groups with optional filtering
        if args.width and args.gate_count:
            dim_group = db.get_dim_group_by_dimensions(args.width, args.gate_count)
            dim_groups = [dim_group] if dim_group else []
        elif args.width:
            dim_groups = db.get_dim_groups_by_width(args.width)
        else:
            dim_groups = db.get_all_dim_groups()

        if not dim_groups:
            print("No dimension groups found.")
            return 0

        # Display dimension groups
        output = format_dimension_groups(
            dim_groups, args.show_representatives, args.show_equivalents, args.format
        )
        print(output)

        # Show representatives if requested
        if args.show_representatives:
            print("\nRepresentatives:")
            print("=" * 80)
            for dg in dim_groups:
                representatives = db.get_representatives_for_dim_group(dg.id)
                if representatives:
                    print(f"\nDimension Group {dg.id} ({dg.width}, {dg.gate_count}):")
                    for rep in representatives:
                        primary_str = " (PRIMARY)" if rep.is_primary else ""
                        print(
                            f"  Rep {rep.id}: Circuit {rep.circuit_id}, Composition: {rep.gate_composition}{primary_str}"
                        )

        # Show equivalents count if requested
        if args.show_equivalents:
            print("\nEquivalents:")
            print("=" * 80)
            for dg in dim_groups:
                equivalents = db.get_all_equivalents_for_dim_group(dg.id)
                print(f"Dimension Group {dg.id}: {len(equivalents)} equivalents")

    except Exception as e:
        logger.error(f"List failed: {e}")
        print(f"✗ List failed: {e}")
        return 1

    return 0


def cmd_unroll(args, factory: IdentityFactory) -> int:
    """Handle unroll command."""
    try:
        if args.all:
            logger.info("Unrolling all unprocessed dimension groups")
            results = factory.unroller.unroll_all_dimension_groups(args.types)

            successful = sum(1 for r in results.values() if r.success)
            print(f"✓ Unrolled {successful}/{len(results)} dimension groups")

            for dim_group_id, result in results.items():
                if result.success:
                    print(
                        f"  Group {dim_group_id}: {result.total_equivalents} equivalents"
                    )
                else:
                    print(f"  Group {dim_group_id}: Failed - {result.error_message}")

        elif args.dim_group_id:
            logger.info(f"Unrolling dimension group {args.dim_group_id}")
            result = factory.unroller.unroll_dimension_group(
                args.dim_group_id, args.types
            )

            if result.success:
                print(f"✓ Unrolled dimension group {args.dim_group_id}")
                print(f"  Total equivalents: {result.total_equivalents}")
                print(f"  New circuits: {result.new_circuits}")
                if result.unroll_types:
                    print("  Unroll types used:")
                    for unroll_type, count in result.unroll_types.items():
                        print(f"    {unroll_type}: {count}")
            else:
                print(f"✗ Unroll failed: {result.error_message}")
                return 1
        else:
            print("Error: Must specify either --all or a dimension group ID")
            return 1

    except Exception as e:
        logger.error(f"Unroll failed: {e}")
        print(f"✗ Unroll failed: {e}")
        return 1

    return 0


def cmd_stats(args, factory: IdentityFactory) -> int:
    """Handle stats command."""
    try:
        stats = factory.get_factory_stats()

        print("Factory Statistics:")
        print("=" * 50)
        print(f"Dimension Groups: {stats.total_dim_groups}")
        print(f"Total Circuits: {stats.total_circuits}")
        print(f"Representatives: {stats.total_representatives}")
        print(f"Equivalents: {stats.total_equivalents}")
        print(f"Generation Time: {stats.generation_time:.2f}s")
        print(f"Unroll Time: {stats.unroll_time:.2f}s")

        if args.detailed:
            print("\nDetailed Statistics:")
            print("-" * 30)
            print(f"Simplifications: {stats.total_simplifications}")
            print(f"Debris Analyses: {stats.total_debris_analyses}")
            print(f"ML Analyses: {stats.total_ml_analyses}")
            print(f"Active Jobs: {stats.active_jobs}")
            print(f"Post-process Time: {stats.post_process_time:.2f}s")
            print(f"Debris Analysis Time: {stats.debris_analysis_time:.2f}s")
            print(f"ML Analysis Time: {stats.ml_analysis_time:.2f}s")

    except Exception as e:
        logger.error(f"Stats failed: {e}")
        print(f"✗ Stats failed: {e}")
        return 1

    return 0


def cmd_analyze(args, factory: IdentityFactory) -> int:
    """Handle analyze command."""
    try:
        analysis = factory.get_dimension_group_analysis(args.dim_group_id)

        if "error" in analysis:
            print(f"✗ Analysis failed: {analysis['error']}")
            return 1

        print(f"Analysis for Dimension Group {args.dim_group_id}:")
        print("=" * 60)
        print(f"Dimensions: ({analysis['width']}, {analysis['gate_count']})")
        print(f"Circuit Count: {analysis['circuit_count']}")
        print(f"Total Equivalents: {analysis['total_equivalents']}")
        print(f"Processed: {'Yes' if analysis['is_processed'] else 'No'}")

        print(f"\nRepresentatives ({len(analysis['representatives'])}):")
        for rep in analysis["representatives"]:
            primary_str = " (PRIMARY)" if rep["is_primary"] else ""
            print(
                f"  {rep['id']}: Circuit {rep['circuit_id']}, Composition: {rep['gate_composition']}{primary_str}"
            )

        if analysis["equivalents"]["by_unroll_type"]:
            print("\nEquivalents by Unroll Type:")
            for unroll_type, count in analysis["equivalents"]["by_unroll_type"].items():
                print(f"  {unroll_type}: {count}")

        if args.export:
            with open(args.export, "w") as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"\nAnalysis exported to: {args.export}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"✗ Analysis failed: {e}")
        return 1

    return 0


def cmd_batch(args, factory: IdentityFactory) -> int:
    """Handle batch command."""
    try:
        # Load config file
        with open(args.config_file, "r") as f:
            config = json.load(f)

        dimensions = []
        if "dimensions" in config:
            dimensions = [(d["width"], d["gate_count"]) for d in config["dimensions"]]
        else:
            print("Error: Config file must contain 'dimensions' array")
            return 1

        print(f"Batch generating {len(dimensions)} dimension groups...")

        results = factory.batch_generate(
            dimensions, max_inverse_gates=args.max_inverse_gates
        )

        successful = sum(1 for r in results.values() if r["success"])
        print(f"✓ Generated {successful}/{len(results)} circuits")

        # Save results
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)

            results_file = output_dir / "batch_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to: {results_file}")

    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        print(f"✗ Batch generation failed: {e}")
        return 1

    return 0


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Set up logging
    setup_logging(args.log_level)

    try:
        # Create factory with config
        config = FactoryConfig(db_path=args.db_path, log_level=args.log_level)
        factory = IdentityFactory(config)

        # Dispatch to command handlers
        if args.command == "generate":
            return cmd_generate(args, factory)
        elif args.command == "list":
            return cmd_list(args, factory)
        elif args.command == "unroll":
            return cmd_unroll(args, factory)
        elif args.command == "stats":
            return cmd_stats(args, factory)
        elif args.command == "analyze":
            return cmd_analyze(args, factory)
        elif args.command == "batch":
            return cmd_batch(args, factory)
        else:
            print(f"Unknown command: {args.command}")
            return 1

    except Exception as e:
        logger.error(f"CLI failed: {e}")
        print(f"✗ CLI failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
