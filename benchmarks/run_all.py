"""
PARAGON GRAND UNIFIED TEST
===========================

Runs all protocol certification tests and generates a summary report.
This is the final gate before deployment.

Usage:
    python -m benchmarks.run_all
    python -m benchmarks.run_all --verbose
    python -m benchmarks.run_all --protocol alpha  # Run specific protocol

Exit Codes:
    0 - All protocols passed
    1 - One or more protocols failed
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class ProtocolStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass
class ProtocolResult:
    """Result from a single protocol run."""
    name: str
    status: ProtocolStatus
    duration_seconds: float
    tests_passed: int
    tests_total: int
    error_message: str = ""


def run_protocol_alpha_wrapper() -> ProtocolResult:
    """Protocol Alpha: Core Graph Operations (Rustworkx)."""
    start = time.perf_counter()
    try:
        from benchmarks.protocol_alpha import run_protocol_alpha
        results = run_protocol_alpha()
        duration = time.perf_counter() - start

        # Results is a list of BenchmarkResult objects
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        status = ProtocolStatus.PASSED if passed == total else ProtocolStatus.FAILED

        return ProtocolResult(
            name="Alpha (Graph Core)",
            status=status,
            duration_seconds=duration,
            tests_passed=passed,
            tests_total=total,
        )
    except Exception as e:
        duration = time.perf_counter() - start
        return ProtocolResult(
            name="Alpha (Graph Core)",
            status=ProtocolStatus.ERROR,
            duration_seconds=duration,
            tests_passed=0,
            tests_total=0,
            error_message=str(e),
        )


def run_protocol_beta_wrapper() -> ProtocolResult:
    """Protocol Beta: Wavefront Traversal (rx.layers)."""
    start = time.perf_counter()
    try:
        from benchmarks.protocol_beta import run_protocol_beta
        results = run_protocol_beta()
        duration = time.perf_counter() - start

        # Results is a list of IntegrityResult objects
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        status = ProtocolStatus.PASSED if passed == total else ProtocolStatus.FAILED

        return ProtocolResult(
            name="Beta (Wavefront)",
            status=status,
            duration_seconds=duration,
            tests_passed=passed,
            tests_total=total,
        )
    except Exception as e:
        duration = time.perf_counter() - start
        return ProtocolResult(
            name="Beta (Wavefront)",
            status=ProtocolStatus.ERROR,
            duration_seconds=duration,
            tests_passed=0,
            tests_total=0,
            error_message=str(e),
        )


def run_protocol_gamma_wrapper() -> ProtocolResult:
    """Protocol Gamma: Schema Validation (msgspec + Pandera)."""
    start = time.perf_counter()
    try:
        from benchmarks.protocol_gamma import run_tests
        # run_tests returns (passed, total, errors)
        passed, total, errors = run_tests()
        duration = time.perf_counter() - start

        status = ProtocolStatus.PASSED if passed == total else ProtocolStatus.FAILED

        return ProtocolResult(
            name="Gamma (Schema)",
            status=status,
            duration_seconds=duration,
            tests_passed=passed,
            tests_total=total,
        )
    except Exception as e:
        duration = time.perf_counter() - start
        return ProtocolResult(
            name="Gamma (Schema)",
            status=ProtocolStatus.ERROR,
            duration_seconds=duration,
            tests_passed=0,
            tests_total=0,
            error_message=str(e),
        )


def run_protocol_delta_wrapper() -> ProtocolResult:
    """Protocol Delta: Graph Alignment (pygmtools)."""
    start = time.perf_counter()
    try:
        from benchmarks.protocol_delta import run_tests
        # run_tests returns (passed, total, errors)
        passed, total, errors = run_tests()
        duration = time.perf_counter() - start

        status = ProtocolStatus.PASSED if passed == total else ProtocolStatus.FAILED

        return ProtocolResult(
            name="Delta (Alignment)",
            status=status,
            duration_seconds=duration,
            tests_passed=passed,
            tests_total=total,
        )
    except Exception as e:
        duration = time.perf_counter() - start
        return ProtocolResult(
            name="Delta (Alignment)",
            status=ProtocolStatus.ERROR,
            duration_seconds=duration,
            tests_passed=0,
            tests_total=0,
            error_message=str(e),
        )


def run_protocol_epsilon_wrapper() -> ProtocolResult:
    """Protocol Epsilon: End-to-End Integration."""
    start = time.perf_counter()
    try:
        from benchmarks.protocol_epsilon import run_protocol_epsilon
        # run_protocol_epsilon returns dict with 'passed', 'total', 'errors'
        result = run_protocol_epsilon()
        duration = time.perf_counter() - start

        passed = result.get("passed", 0)
        total = result.get("total", 0)
        status = ProtocolStatus.PASSED if passed == total else ProtocolStatus.FAILED

        return ProtocolResult(
            name="Epsilon (Integration)",
            status=status,
            duration_seconds=duration,
            tests_passed=passed,
            tests_total=total,
        )
    except Exception as e:
        duration = time.perf_counter() - start
        return ProtocolResult(
            name="Epsilon (Integration)",
            status=ProtocolStatus.ERROR,
            duration_seconds=duration,
            tests_passed=0,
            tests_total=0,
            error_message=str(e),
        )


def run_protocol_zeta_wrapper() -> ProtocolResult:
    """Protocol Zeta: Human-in-the-Loop & Research Phase."""
    start = time.perf_counter()
    try:
        from benchmarks.protocol_zeta import run_protocol_zeta
        # run_protocol_zeta returns dict with 'passed', 'total', 'errors'
        result = run_protocol_zeta()
        duration = time.perf_counter() - start

        passed = result.get("passed", 0)
        total = result.get("total", 0)
        status = ProtocolStatus.PASSED if passed == total else ProtocolStatus.FAILED

        return ProtocolResult(
            name="Zeta (Research)",
            status=status,
            duration_seconds=duration,
            tests_passed=passed,
            tests_total=total,
        )
    except Exception as e:
        duration = time.perf_counter() - start
        return ProtocolResult(
            name="Zeta (Research)",
            status=ProtocolStatus.ERROR,
            duration_seconds=duration,
            tests_passed=0,
            tests_total=0,
            error_message=str(e),
        )


def run_protocol_omega_wrapper() -> ProtocolResult:
    """Protocol Omega: Full System Stress Test (via Harness)."""
    start = time.perf_counter()
    try:
        from benchmarks.harness import run_benchmark
        # Run smoke tier as default for quick validation
        result = run_benchmark(tier="smoke", interactive=False, visualize=False)
        duration = time.perf_counter() - start

        passed = result.tasks_passed
        total = result.tasks_attempted
        status = ProtocolStatus.PASSED if passed == total else ProtocolStatus.FAILED

        return ProtocolResult(
            name="Omega (Harness)",
            status=status,
            duration_seconds=duration,
            tests_passed=passed,
            tests_total=total,
        )
    except Exception as e:
        duration = time.perf_counter() - start
        return ProtocolResult(
            name="Omega (Harness)",
            status=ProtocolStatus.ERROR,
            duration_seconds=duration,
            tests_passed=0,
            tests_total=0,
            error_message=str(e),
        )


PROTOCOL_MAP = {
    "alpha": run_protocol_alpha_wrapper,
    "beta": run_protocol_beta_wrapper,
    "gamma": run_protocol_gamma_wrapper,
    "delta": run_protocol_delta_wrapper,
    "epsilon": run_protocol_epsilon_wrapper,
    "zeta": run_protocol_zeta_wrapper,
    "omega": run_protocol_omega_wrapper,
}


def print_header():
    """Print the certification header."""
    print("=" * 70)
    print("  PARAGON CERTIFICATION SUITE")
    print("  Grand Unified Test - All Protocols")
    print("=" * 70)
    print()


def print_result(result: ProtocolResult, verbose: bool = False):
    """Print a single protocol result."""
    status_symbol = {
        ProtocolStatus.PASSED: "[PASS]",
        ProtocolStatus.FAILED: "[FAIL]",
        ProtocolStatus.SKIPPED: "[SKIP]",
        ProtocolStatus.ERROR: "[ERR!]",
    }

    symbol = status_symbol[result.status]
    test_info = f"({result.tests_passed}/{result.tests_total})"
    duration = f"{result.duration_seconds:.3f}s"

    print(f"  {symbol} {result.name:<25} {test_info:<10} {duration:>8}")

    if verbose and result.error_message:
        print(f"         Error: {result.error_message[:60]}...")


def print_summary(results: list[ProtocolResult], total_duration: float):
    """Print the final summary."""
    print()
    print("-" * 70)

    total_tests = sum(r.tests_total for r in results)
    passed_tests = sum(r.tests_passed for r in results)

    passed_protocols = sum(1 for r in results if r.status == ProtocolStatus.PASSED)
    total_protocols = len(results)

    all_passed = passed_protocols == total_protocols

    print(f"  PROTOCOLS: {passed_protocols}/{total_protocols} passed")
    print(f"  TESTS:     {passed_tests}/{total_tests} passed")
    print(f"  DURATION:  {total_duration:.3f}s")
    print()

    if all_passed:
        print("  CERTIFICATION: APPROVED")
        print("  Status: Ready for deployment")
    else:
        print("  CERTIFICATION: DENIED")
        print("  Status: Fix failures before deployment")

    print("=" * 70)

    return all_passed


def main():
    """Run all certification protocols."""
    parser = argparse.ArgumentParser(description="Paragon Certification Suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--protocol", "-p", choices=list(PROTOCOL_MAP.keys()),
                        help="Run a specific protocol only")
    args = parser.parse_args()

    print_header()

    # Determine which protocols to run
    if args.protocol:
        protocols_to_run = {args.protocol: PROTOCOL_MAP[args.protocol]}
    else:
        protocols_to_run = PROTOCOL_MAP

    results = []
    total_start = time.perf_counter()

    for name, runner in protocols_to_run.items():
        print(f"  Running Protocol {name.capitalize()}...", end="", flush=True)
        result = runner()
        results.append(result)
        print(f"\r", end="")  # Clear the "Running..." line
        print_result(result, args.verbose)

    total_duration = time.perf_counter() - total_start

    all_passed = print_summary(results, total_duration)

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
