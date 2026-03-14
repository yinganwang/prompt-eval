#!/usr/bin/env python3
"""Evaluation harness for customer support ticket classification prompts.

Runs each prompt variant against all labeled tickets, scores accuracy,
and saves detailed results to the results/ directory.
"""

import argparse
import asyncio
import json

from dotenv import load_dotenv
load_dotenv()
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from prompts import CATEGORIES, PROMPT_VARIANTS

MODEL = "claude-haiku-4-5-20251001"
MAX_CONCURRENT = 10  # Limit concurrent API calls to avoid rate limits.
MAX_RETRIES = 5
BASE_DELAY = 1.0  # Seconds for exponential backoff.

# Rough pricing for claude-haiku-4-5 (per 1M tokens).
INPUT_COST_PER_M = 1.00
OUTPUT_COST_PER_M = 5.00

RESULTS_DIR = Path(__file__).parent / "results"
DATA_PATH = Path(__file__).parent / "data" / "tickets.json"


def load_tickets(path: Path = DATA_PATH) -> list[dict]:
    """Load labeled tickets from JSON file."""
    with open(path) as f:
        tickets = json.load(f)
    for t in tickets:
        if t["label"] not in CATEGORIES:
            raise ValueError(f"Unknown label {t['label']!r} in ticket: {t['text'][:60]}")
    return tickets


def parse_label(response_text: str) -> str | None:
    """Extract a category label from a model response.

    Handles varied formats:
      - bare label: "billing"
      - sentence: "The category is billing."
      - CoT with ANSWER: "...reasoning...\\nANSWER: billing"
      - JSON: {"category": "billing"}
      - Uppercase/mixed case: "BILLING", "Feature_Request"
    """
    text = response_text.strip()

    # Try JSON parse first (for prompt_v4_structured).
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "category" in data:
            candidate = data["category"].strip().lower()
            if candidate in CATEGORIES:
                return candidate
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try extracting JSON from within the response (sometimes wrapped in markdown).
    json_match = re.search(r'\{[^}]*"category"\s*:\s*"([^"]+)"[^}]*\}', text)
    if json_match:
        candidate = json_match.group(1).strip().lower()
        if candidate in CATEGORIES:
            return candidate

    # Try ANSWER: pattern (for CoT prompts).
    answer_match = re.search(r"ANSWER:\s*(\S+)", text, re.IGNORECASE)
    if answer_match:
        candidate = answer_match.group(1).strip().lower().rstrip(".")
        if candidate in CATEGORIES:
            return candidate

    # Try to find any category mentioned in the text — search the tail first
    # (last 200 chars) where the conclusion typically lives, to avoid picking
    # up category names mentioned mid-reasoning in CoT responses.
    text_lower = text.lower()
    tail = text_lower[-200:] if len(text_lower) > 200 else text_lower
    for search_region in (tail, text_lower):
        found = []
        for cat in CATEGORIES:
            # Use word boundary matching to avoid partial matches.
            for m in re.finditer(rf"\b{re.escape(cat)}\b", search_region):
                found.append((m.start(), cat))
        if found:
            found.sort(key=lambda x: x[0])
            return found[-1][1]
        # If we already searched the full text (tail == text_lower), stop.
        if search_region is text_lower:
            break

    return None


async def classify_ticket(
    client: anthropic.AsyncAnthropic,
    prompt_fn: callable,
    ticket_text: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Send a single ticket to the API and return the result."""
    prompt = prompt_fn(ticket_text)
    messages = [{"role": "user", "content": prompt["user"]}]
    kwargs = {
        "model": MODEL,
        "max_tokens": 300,
        "messages": messages,
    }
    if prompt.get("system"):
        kwargs["system"] = prompt["system"]

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await client.messages.create(**kwargs)
            raw_text = response.content[0].text
            predicted = parse_label(raw_text)
            return {
                "raw_response": raw_text,
                "predicted_label": predicted,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "error": None,
            }
        except anthropic.RateLimitError as e:
            last_error = str(e)
            delay = BASE_DELAY * (2 ** attempt)
            await asyncio.sleep(delay)
        except anthropic.APIStatusError as e:
            last_error = str(e)
            if e.status_code >= 500:
                delay = BASE_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)
            else:
                break
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            delay = BASE_DELAY * (2 ** attempt)
            await asyncio.sleep(delay)

    return {
        "raw_response": None,
        "predicted_label": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "error": last_error,
    }


async def evaluate_prompt(
    client: anthropic.AsyncAnthropic,
    prompt_name: str,
    prompt_fn: callable,
    tickets: list[dict],
) -> dict:
    """Evaluate a single prompt variant against all tickets."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [
        classify_ticket(client, prompt_fn, t["text"], semaphore)
        for t in tickets
    ]
    results = await asyncio.gather(*tasks)

    # Score results.
    correct = 0
    total = len(tickets)
    category_stats: dict[str, dict] = {
        cat: {"correct": 0, "total": 0} for cat in CATEGORIES
    }
    details = []
    total_input_tokens = 0
    total_output_tokens = 0

    for ticket, result in zip(tickets, results):
        true_label = ticket["label"]
        pred_label = result["predicted_label"]
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1

        category_stats[true_label]["total"] += 1
        if is_correct:
            category_stats[true_label]["correct"] += 1

        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]

        details.append({
            "text": ticket["text"],
            "true_label": true_label,
            "predicted_label": pred_label,
            "correct": is_correct,
            "raw_response": result["raw_response"],
            "error": result["error"],
        })

    accuracy = correct / total if total > 0 else 0.0
    per_category = {}
    for cat, stats in category_stats.items():
        cat_total = stats["total"]
        per_category[cat] = {
            "correct": stats["correct"],
            "total": cat_total,
            "accuracy": stats["correct"] / cat_total if cat_total > 0 else 0.0,
        }

    cost_estimate = (
        (total_input_tokens / 1_000_000) * INPUT_COST_PER_M
        + (total_output_tokens / 1_000_000) * OUTPUT_COST_PER_M
    )

    return {
        "prompt_name": prompt_name,
        "model": MODEL,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_category": per_category,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "cost_estimate_usd": round(cost_estimate, 6),
        "details": details,
    }


_CATEGORY_DISPLAY = {
    "feature_request": "feat_req",
    "general": "general",
    "billing": "billing",
    "technical": "tech",
    "account": "account",
}


def print_summary(results: list[dict], show_std: bool = False) -> None:
    """Print a formatted summary table of evaluation results."""
    col_width = 10
    acc_col = f"{'Accuracy (±std)' if show_std else 'Accuracy':>16}"
    header = f"{'Prompt':<25} {acc_col}  "
    for cat in CATEGORIES:
        display = _CATEGORY_DISPLAY.get(cat, cat[:col_width])
        header += f"{display:>{col_width}}  "
    header += f"{'Cost ($)':>8}"
    print("\n" + "=" * len(header))
    print("EVALUATION RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    best_accuracy = -1.0
    best_prompt = None

    for r in results:
        if show_std and "accuracy_std" in r:
            acc_str = f"{r['accuracy']:>7.1%} ±{r['accuracy_std']:.1%}"
        else:
            acc_str = f"{r['accuracy']:>7.1%}"
        line = f"{r['prompt_name']:<25} {acc_str:>16}  "
        for cat in CATEGORIES:
            cat_acc = r["per_category"][cat]["accuracy"]
            line += f"{cat_acc:>{col_width}.1%}  "
        line += f"${r['cost_estimate_usd']:>7.4f}"
        print(line)

        if r["accuracy"] > best_accuracy:
            best_accuracy = r["accuracy"]
            best_prompt = r["prompt_name"]

    print("-" * len(header))
    print(f"\nWinner: {best_prompt} ({best_accuracy:.1%} accuracy)")


def average_results(runs: list[list[dict]]) -> list[dict]:
    """Average accuracy and cost across multiple runs of the same prompts."""
    averaged = []
    for prompt_idx in range(len(runs[0])):
        base = runs[0][prompt_idx]
        n = len(runs)

        avg_accuracy = sum(r[prompt_idx]["accuracy"] for r in runs) / n
        std_accuracy = (
            sum((r[prompt_idx]["accuracy"] - avg_accuracy) ** 2 for r in runs) / n
        ) ** 0.5
        avg_cost = sum(r[prompt_idx]["cost_estimate_usd"] for r in runs) / n

        per_category: dict[str, dict] = {}
        for cat in CATEGORIES:
            cat_acc_values = [r[prompt_idx]["per_category"][cat]["accuracy"] for r in runs]
            per_category[cat] = {
                "correct": runs[-1][prompt_idx]["per_category"][cat]["correct"],
                "total": base["per_category"][cat]["total"],
                "accuracy": sum(cat_acc_values) / n,
            }

        averaged.append({
            **base,
            "accuracy": avg_accuracy,
            "accuracy_std": std_accuracy,
            "cost_estimate_usd": avg_cost,
            "per_category": per_category,
            "num_runs": n,
        })
    return averaged


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate prompt variants for ticket classification."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only 1 ticket per category (5 total) for quick testing.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="Run the full eval N times and report mean ± std accuracy (default: 1).",
    )
    args = parser.parse_args()

    tickets = load_tickets()
    if args.dry_run:
        # Take 1 per category to keep it representative.
        seen: dict[str, int] = {}
        sampled = []
        for t in tickets:
            count = seen.get(t["label"], 0)
            if count < 1:
                sampled.append(t)
                seen[t["label"]] = count + 1
            if len(sampled) >= len(CATEGORIES):
                break
        tickets = sampled
        print(f"Dry-run mode: using {len(tickets)} tickets")

    repeat = max(1, args.repeat)
    print(f"Evaluating {len(PROMPT_VARIANTS)} prompts × {len(tickets)} tickets"
          + (f" × {repeat} runs" if repeat > 1 else ""))
    print(f"Model: {MODEL}\n")

    client = anthropic.AsyncAnthropic()
    all_runs: list[list[dict]] = []

    for run_idx in range(repeat):
        if repeat > 1:
            print(f"── Run {run_idx + 1}/{repeat} ──")
        run_results = []
        for name, fn in PROMPT_VARIANTS.items():
            print(f"Running {name}...", end=" ", flush=True)
            start = time.monotonic()
            result = await evaluate_prompt(client, name, fn, tickets)
            elapsed = time.monotonic() - start
            errors = sum(1 for d in result["details"] if d["error"])
            parse_failures = sum(
                1 for d in result["details"]
                if d["predicted_label"] is None and d["error"] is None
            )
            print(
                f"{result['accuracy']:.1%} "
                f"({elapsed:.1f}s, {errors} errors, {parse_failures} parse failures)"
            )
            run_results.append(result)
        all_runs.append(run_results)

    all_results = average_results(all_runs) if repeat > 1 else all_runs[0]
    print_summary(all_results, show_std=repeat > 1)

    # Save results.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"run_{timestamp}.json"
    output = {
        "timestamp": timestamp,
        "model": MODEL,
        "num_tickets": len(tickets),
        "dry_run": args.dry_run,
        "num_runs": repeat,
        "results": all_results if isinstance(all_results, list) else [all_results],
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
