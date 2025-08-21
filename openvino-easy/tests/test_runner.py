#!/usr/bin/env python3
"""Test runner for OpenVINO-Easy with different test modes."""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description, project_root):
    """Run a command and handle output."""
    print(f"\n[TEST] {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)

    try:
        subprocess.run(cmd, check=True, cwd=project_root)
        print(f"[PASS] {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {description} - FAILED (exit code: {e.returncode})")
        return False


def main():
    # Force UTF-8 encoding for console output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # Find project root (parent of tests directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(description="OpenVINO-Easy test runner")
    parser.add_argument(
        "--mode",
        choices=[
            "fast",
            "full",
            "unit",
            "integration",
            "e2e",
            "coverage",
            "performance",
            "audio",
        ],
        default="fast",
        help="Test mode to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--no-capture",
        "-s",
        action="store_true",
        help="Don't capture output (pytest -s)",
    )

    args = parser.parse_args()

    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]

    if args.verbose:
        base_cmd.append("-v")

    if args.no_capture:
        base_cmd.append("-s")

    success = True

    if args.mode == "fast":
        # Run only fast unit tests
        cmd = base_cmd + ["tests/", "-m", "not slow and not integration"]
        success &= run_command(cmd, "Fast unit tests", project_root)

    elif args.mode == "unit":
        # Run all unit tests (including slow ones)
        cmd = base_cmd + ["tests/", "-m", "not integration"]
        success &= run_command(cmd, "All unit tests", project_root)

    elif args.mode == "integration":
        # Run integration tests only
        cmd = base_cmd + ["tests/", "-m", "integration and not e2e"]
        success &= run_command(cmd, "Integration tests", project_root)

    elif args.mode == "e2e":
        # Run end-to-end tests with real models
        cmd = base_cmd + ["tests/test_e2e_real_models.py", "-m", "not slow"]
        success &= run_command(cmd, "End-to-end tests (fast)", project_root)

        # Run slow e2e tests separately
        cmd = base_cmd + ["tests/test_e2e_real_models.py", "-m", "slow"]
        success &= run_command(cmd, "End-to-end tests (slow)", project_root)

    elif args.mode == "performance":
        # Run performance regression tests
        cmd = base_cmd + ["tests/test_performance_regression.py"]
        success &= run_command(cmd, "Performance regression tests", project_root)

    elif args.mode == "audio":
        # Run audio-specific tests
        cmd = base_cmd + [
            "tests/test_audio.py",
            "tests/test_e2e_real_models.py::TestE2ERealModels::test_audio_model_e2e",
        ]
        success &= run_command(cmd, "Audio functionality tests", project_root)

    elif args.mode == "full":
        # Run everything except very slow tests
        cmd = base_cmd + ["tests/", "-m", "not slow"]
        success &= run_command(cmd, "All tests (excluding slow)", project_root)

    elif args.mode == "coverage":
        # Run with coverage
        cmd = base_cmd + [
            "--cov=oe",
            "--cov-report=html",
            "--cov-report=term",
            "tests/",
            "-m",
            "not slow",
        ]
        success &= run_command(cmd, "Tests with coverage", project_root)

        if success:
            print("\n[INFO] Coverage report generated in htmlcov/index.html")

    # Summary
    print("\n" + "=" * 50)
    if success:
        print("[SUCCESS] All tests PASSED!")
        sys.exit(0)
    else:
        print("[ERROR] Some tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
