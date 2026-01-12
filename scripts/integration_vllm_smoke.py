#!/usr/bin/env python3
"""
Smoke test script for vLLM integration.

Validates that the LLM backend produces valid JSON responses within
acceptable time limits for canonical DICOM NL queries.

Usage:
    # Using default config (configs/llm.yaml)
    python scripts/integration_vllm_smoke.py

    # With custom config
    python scripts/integration_vllm_smoke.py --config configs/llm-test.yaml

    # With custom timeout threshold
    python scripts/integration_vllm_smoke.py --max-time 15

    # Verbose output (show LLM responses)
    python scripts/integration_vllm_smoke.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dicom_nlquery.llm_client import create_llm_client
from dicom_nlquery.models import LLMConfig
from dicom_nlquery.nl_parser import extract_json, SYSTEM_PROMPT

import yaml


@dataclass
class TestCase:
    """A smoke test case with query and expected fields."""

    name: str
    query: str
    expected_fields: dict[str, list[str]]  # path -> expected values (any match)
    must_have_fields: list[str]  # fields that must be present and non-null


@dataclass
class TestResult:
    """Result of a single test case."""

    name: str
    passed: bool
    duration_seconds: float
    error: str | None = None
    response: dict[str, Any] | None = None
    warnings: list[str] | None = None


# Canonical test cases covering key scenarios
TEST_CASES = [
    TestCase(
        name="CT angiogram with routing and age",
        query="studies from year 2000 until 2022 of CT chest angiograms from ORTHANC to RADIANT, patients age 20 to 80",
        expected_fields={
            "study.modality_in_study": ["CT"],
        },
        must_have_fields=["study.study_description", "study.study_date", "study.patient_birth_date"],
    ),
    TestCase(
        name="MRI with sex filter",
        query="cranial MRI for women ages 30 to 50",
        expected_fields={
            "study.modality_in_study": ["MR"],
            "study.patient_sex": ["F"],
        },
        must_have_fields=["study.patient_birth_date"],
    ),
    TestCase(
        name="Ultrasound with body part",
        query="obstetric ultrasound exams from last month",
        expected_fields={
            "study.modality_in_study": ["US"],
        },
        must_have_fields=["study.study_description"],
    ),
    TestCase(
        name="Simple modality query",
        query="CT exams",
        expected_fields={
            "study.modality_in_study": ["CT"],
        },
        must_have_fields=[],
    ),
    TestCase(
        name="Patient name query",
        query="exams for patient SILVA",
        expected_fields={},
        must_have_fields=["study.patient_name"],
    ),
]


def load_llm_config(config_path: str) -> LLMConfig:
    """Load LLM configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        # Try relative to project root
        alt_path = Path(__file__).parent.parent.parent / config_path
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return LLMConfig.model_validate(data)


def get_nested_value(data: dict, path: str) -> Any:
    """Get a nested value from a dict using dot notation."""
    keys = path.split(".")
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def run_test_case(
    client: Any,
    test: TestCase,
    max_time: float,
    verbose: bool = False,
) -> TestResult:
    """Run a single test case and return the result."""
    warnings: list[str] = []

    # Build prompt with today's date for age calculations
    from datetime import date

    system_prompt = f"{SYSTEM_PROMPT}\nToday: {date.today():%Y-%m-%d}"

    start = time.perf_counter()
    try:
        raw_response = client.chat(system_prompt, test.query, json_mode=True)
        duration = time.perf_counter() - start

        if verbose:
            print(f"    Raw response ({len(raw_response)} chars): {raw_response[:500]}...")

        # Parse JSON
        try:
            data = extract_json(raw_response)
        except ValueError as e:
            return TestResult(
                name=test.name,
                passed=False,
                duration_seconds=duration,
                error=f"Invalid JSON: {e}",
            )

        # Check time limit
        if duration > max_time:
            warnings.append(f"Slow response: {duration:.2f}s > {max_time}s")

        # Check required fields
        for field_path in test.must_have_fields:
            value = get_nested_value(data, field_path)
            if value is None or value == "":
                return TestResult(
                    name=test.name,
                    passed=False,
                    duration_seconds=duration,
                    error=f"Missing required field: {field_path}",
                    response=data,
                )

        # Check expected field values
        for field_path, expected_values in test.expected_fields.items():
            value = get_nested_value(data, field_path)
            if value is None:
                return TestResult(
                    name=test.name,
                    passed=False,
                    duration_seconds=duration,
                    error=f"Missing expected field: {field_path}",
                    response=data,
                )
            # Check if value matches any expected value
            value_upper = str(value).upper()
            if not any(exp.upper() in value_upper for exp in expected_values):
                warnings.append(
                    f"Field {field_path}={value} doesn't match expected {expected_values}"
                )

        # Validate basic structure
        if not isinstance(data.get("study"), dict):
            return TestResult(
                name=test.name,
                passed=False,
                duration_seconds=duration,
                error="Response missing 'study' object",
                response=data,
            )

        return TestResult(
            name=test.name,
            passed=True,
            duration_seconds=duration,
            response=data,
            warnings=warnings if warnings else None,
        )

    except Exception as e:
        duration = time.perf_counter() - start
        return TestResult(
            name=test.name,
            passed=False,
            duration_seconds=duration,
            error=f"Exception: {type(e).__name__}: {e}",
        )


def print_result(result: TestResult, verbose: bool = False) -> None:
    """Print a test result with formatting."""
    status = "\033[92mPASS\033[0m" if result.passed else "\033[91mFAIL\033[0m"
    time_color = "\033[93m" if result.duration_seconds > 5 else ""
    time_reset = "\033[0m" if time_color else ""

    print(f"  [{status}] {result.name} ({time_color}{result.duration_seconds:.2f}s{time_reset})")

    if result.error:
        print(f"        Error: {result.error}")

    if result.warnings:
        for warning in result.warnings:
            print(f"        Warning: {warning}")

    if verbose and result.response:
        print(f"        Response: {json.dumps(result.response, indent=2)[:500]}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for vLLM/LLM integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/llm.yaml",
        help="Path to LLM config YAML (default: configs/llm.yaml)",
    )
    parser.add_argument(
        "--max-time",
        "-t",
        type=float,
        default=10.0,
        help="Maximum acceptable response time in seconds (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output including LLM responses",
    )
    parser.add_argument(
        "--query",
        "-q",
        help="Run a single custom query instead of test cases",
    )
    args = parser.parse_args()

    # Load config
    try:
        config = load_llm_config(args.config)
    except FileNotFoundError as e:
        print(f"\033[91mError:\033[0m {e}")
        return 1

    print(f"\n\033[1mLLM Smoke Test\033[0m")
    print(f"  Config: {args.config}")
    print(f"  Provider: {config.provider}")
    print(f"  Model: {config.model}")
    print(f"  Base URL: {config.base_url}")
    print(f"  Max time: {args.max_time}s")
    print()

    # Create client
    try:
        client = create_llm_client(config)
    except Exception as e:
        print(f"\033[91mError creating LLM client:\033[0m {e}")
        return 1

    # Run custom query if provided
    if args.query:
        print(f"Running custom query: {args.query}\n")
        test = TestCase(
            name="Custom query",
            query=args.query,
            expected_fields={},
            must_have_fields=[],
        )
        result = run_test_case(client, test, args.max_time, args.verbose)
        print_result(result, verbose=True)
        if result.response:
            print(f"\n  Full response:\n{json.dumps(result.response, indent=2)}")
        return 0 if result.passed else 1

    # Run all test cases
    print(f"Running {len(TEST_CASES)} test cases...\n")

    results: list[TestResult] = []
    for test in TEST_CASES:
        if args.verbose:
            print(f"  Testing: {test.name}")
            print(f"    Query: {test.query}")

        result = run_test_case(client, test, args.max_time, args.verbose)
        results.append(result)
        print_result(result, args.verbose)

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    total_time = sum(r.duration_seconds for r in results)
    avg_time = total_time / len(results) if results else 0

    print()
    print(f"\033[1mSummary:\033[0m")
    print(f"  Passed: {passed}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg time: {avg_time:.2f}s")

    if failed > 0:
        print(f"\n\033[91m{failed} test(s) failed!\033[0m")
        return 1

    warnings_count = sum(len(r.warnings or []) for r in results)
    if warnings_count > 0:
        print(f"\n\033[93m{warnings_count} warning(s)\033[0m")

    print(f"\n\033[92mAll tests passed!\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
