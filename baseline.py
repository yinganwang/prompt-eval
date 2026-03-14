#!/usr/bin/env python3
"""CI-style baseline checker for prompt evaluation.

Compares a new prompt against the best existing result and exits with
appropriate status code for CI integration.
"""

import argparse
import asyncio
import importlib.util

from dotenv import load_dotenv
load_dotenv()
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from eval import (
    RESULTS_DIR,
    evaluate_prompt,
    load_tickets,
)
from prompts import CATEGORIES


def load_best_baseline() -> tuple[dict, str] | None:
    """Load the result with the highest accuracy from the results/ directory.

    Returns a tuple of (best_result_dict, source_file_path), or None if no
    results exist.
    """
    if not RESULTS_DIR.exists():
        return None

    best_result = None
    best_accuracy = -1.0
    best_source_file = None

    for path in sorted(RESULTS_DIR.glob("run_*.json")):
        try:
            with open(path) as f:
                run_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Skip dry runs — they aren't representative baselines.
        if run_data.get("dry_run", False):
            continue

        for result in run_data.get("results", []):
            accuracy = result.get("accuracy", 0.0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_result = result
                best_source_file = str(path)

    if best_result is None:
        return None
    return best_result, best_source_file


def load_prompt_function(prompt_file: str) -> callable:
    """Dynamically load a get_prompt(text) function from a Python file."""
    path = Path(prompt_file).resolve()
    if not path.exists():
        print(f"Error: prompt file not found: {path}", file=sys.stderr)
        sys.exit(2)

    spec = importlib.util.spec_from_file_location("user_prompt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "get_prompt"):
        print(
            f"Error: {path} must define a get_prompt(text) function",
            file=sys.stderr,
        )
        sys.exit(2)

    return module.get_prompt


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a new prompt against the best baseline result."
    )
    parser.add_argument(
        "--prompt-file",
        required=True,
        help="Path to a Python file with a get_prompt(text) function.",
    )
    parser.add_argument(
        "--prompt-name",
        default=None,
        help="Name for the new prompt (default: filename stem).",
    )
    args = parser.parse_args()

    # Load baseline.
    baseline_data = load_best_baseline()
    if baseline_data is None:
        print("No baseline found in results/. Run eval.py first.")
        print("Proceeding without a baseline — will save result as the new baseline.\n")
        baseline_accuracy = 0.0
        baseline_name = "(none)"
    else:
        baseline, source_file = baseline_data
        baseline_accuracy = baseline["accuracy"]
        baseline_name = baseline["prompt_name"]
        print(f"Baseline: {baseline_name} — {baseline_accuracy:.1%} accuracy")
        print(f"  Source: {source_file}\n")

    # Load user prompt.
    prompt_fn = load_prompt_function(args.prompt_file)
    prompt_name = args.prompt_name or Path(args.prompt_file).stem

    # Run evaluation.
    tickets = load_tickets()
    client = anthropic.AsyncAnthropic()

    print(f"Evaluating '{prompt_name}' against {len(tickets)} tickets...")
    start = time.monotonic()
    result = await evaluate_prompt(client, prompt_name, prompt_fn, tickets)
    elapsed = time.monotonic() - start
    new_accuracy = result["accuracy"]

    print(f"Completed in {elapsed:.1f}s\n")

    # Print accuracy and per-category breakdown inline (no summary table,
    # since comparing a single result against a baseline is not a multi-prompt
    # comparison).
    print(f"  Accuracy: {new_accuracy:.1%} ({result['correct']}/{result['total']})")
    print(f"  Cost:     ${result['cost_estimate_usd']:.4f}")
    print("  Per-category:")
    for cat in CATEGORIES:
        cat_stats = result["per_category"][cat]
        print(f"    {cat:<20} {cat_stats['accuracy']:>6.1%} ({cat_stats['correct']}/{cat_stats['total']})")

    # Compare.
    print()
    if new_accuracy >= baseline_accuracy:
        print(
            f"PASS: new prompt {new_accuracy:.1%} "
            f"beats baseline {baseline_accuracy:.1%}"
        )
        passed = True
    else:
        print(
            f"FAIL: new prompt {new_accuracy:.1%} "
            f"does not beat baseline {baseline_accuracy:.1%}"
        )
        passed = False

    # Save result if it passes.
    if passed:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"baseline_{timestamp}.json"
        output = {
            "timestamp": timestamp,
            "model": result["model"],
            "num_tickets": len(tickets),
            "dry_run": False,
            "baseline_comparison": {
                "baseline_prompt": baseline_name,
                "baseline_accuracy": baseline_accuracy,
                "new_prompt": prompt_name,
                "new_accuracy": new_accuracy,
                "passed": True,
            },
            "results": [result],
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Result saved to {output_path}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
