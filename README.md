# Prompt Evaluation System — Customer Support Ticket Classifier

Evaluates 5 prompt variants for classifying support tickets into categories:
`billing`, `technical`, `account`, `feature_request`, `general`.

## Setup

```bash
uv sync
export ANTHROPIC_API_KEY="your-key-here"
```

## Usage

### Run full evaluation

```bash
uv run eval.py
```

This evaluates all 5 prompt variants against 50 labeled tickets using
`claude-haiku-4-5`, prints a summary table, and saves detailed results
to `results/run_{timestamp}.json`.

### Quick test (dry run)

```bash
uv run eval.py --dry-run
```

Runs only 5 tickets per prompt for fast iteration.

### Compare a new prompt against the baseline

Create a Python file with a `get_prompt(text)` function that returns a dict
with an optional `"system"` key and a required `"user"` key:

```python
# my_prompt.py
def get_prompt(text):
    return {
        "user": f"Classify this ticket as billing, technical, account, feature_request, or general: {text}"
    }
```

Then run:

```bash
uv run baseline.py --prompt-file my_prompt.py
```

Exits 0 if the new prompt matches or beats the best existing result, exits 1
otherwise.

## File structure

```
data/tickets.json   — 50 labeled example tickets (10 per category)
prompts.py          — 5 prompt variant functions
eval.py             — Main evaluation harness
baseline.py         — CI-style baseline comparison
results/            — Saved evaluation results (created at runtime)
```
