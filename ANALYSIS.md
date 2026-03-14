# Prompt Evaluation Analysis

## Results Summary

Two full evaluation runs (50 tickets × 5–6 prompts, `claude-haiku-4-5`) produced the following:

### Run 1 (5 prompts)

| Prompt | Accuracy | Cost |
|---|---|---|
| `prompt_v1_basic` | 94% | $0.0041 |
| `prompt_v2_examples` | 94% | $0.0110 |
| `prompt_v3_cot` | **96%** | $0.0590 |
| `prompt_v4_structured` | 94% | $0.0075 |
| `prompt_v5_persona` | **98%** | $0.0095 |

### Run 2 (6 prompts, Tree of Thought added)

| Prompt | Accuracy | Cost |
|---|---|---|
| `prompt_v1_basic` | 96% | $0.0041 |
| `prompt_v2_examples` | 96% | $0.0110 |
| `prompt_v3_cot` | 94% | $0.0577 |
| `prompt_v4_structured` | 94% | $0.0075 |
| `prompt_v5_persona` | **98%** | $0.0095 |
| `prompt_v6_tot` | 86% | $0.0825 |

**Overall winner across both runs: `prompt_v5_persona` at 98%.**

---

## Why Accuracy Is Inconsistent Between Runs

Comparing the two runs, several prompts shifted by ±2%:

| Prompt | Run 1 | Run 2 | Δ |
|---|---|---|---|
| `v1_basic` | 94% | 96% | +1 ticket |
| `v2_examples` | 94% | 96% | +1 ticket |
| `v3_cot` | 96% | 94% | −1 ticket |
| `v4_structured` | 94% | 94% | — |
| `v5_persona` | 98% | 98% | — |

Every swing is exactly **±1 ticket** out of 50. With a 50-example dataset, 1 ticket = 2% accuracy — so what looks like a meaningful gap is a single borderline prediction flipping.

### Root Cause: LLM Non-Determinism

LLMs run with `temperature > 0` by default, so the same prompt + input does not always produce the same output. On ambiguous tickets that sit close to a category boundary, the model occasionally flips its prediction between runs. That randomness shows up as accuracy variance across evaluation runs.

---

## Lessons Learned

### 1. Small datasets make results unreliable

With 50 tickets, a 94% vs 96% difference is literally 1 ticket. That is not a meaningful signal — it is noise. You need ~500+ labeled examples before single-ticket swings represent less than 0.5% accuracy change.

### 2. Run each prompt multiple times and average

A single eval run is a **point estimate with no error bars**. Run 5–10 times and report mean ± std deviation. For example, if `v3_cot` scores 94%, 96%, 95%, 94%, 96% across runs, the true accuracy is ~95% ± 1% — far more trustworthy than any single number.

### 3. This is the brittleness problem firsthand

Prompts that look different by 2–4% on a small eval may be statistically indistinguishable. `v5_persona` at 98% is likely genuinely better. But `v1_basic` at 94% vs `v3_cot` at 96% could easily swap on the next run — you cannot rely on that gap to make decisions.

### 4. More reasoning steps ≠ better accuracy

Tree of Thought (`v6_tot`) was the **most expensive** (9× the cost of `v5_persona`) and the **least accurate** (86%). The multi-branch structure introduced more opportunities to go wrong on borderline tickets, particularly in the `account` category (60% accuracy). For a well-scoped classification task, clear category definitions in a system prompt outperformed elaborate reasoning chains.

### 5. The CI baseline needs a margin, not hard equality

The current `baseline.py` passes a new prompt if it ties or beats the saved baseline. But if the baseline was recorded on a lucky run (96%) and you re-run it naturally (94%), you fail CI for a prompt that is actually equivalent. A more robust rule: **require beating the baseline by at least +4%** (2 tickets on a 50-ticket set), or run 3× and compare averages.

### 6. Ambiguous tickets are doing the heavy lifting

The tickets that flip between runs are always the same borderline ones — not random tickets, but the ones that genuinely sit between two categories. Identifying and annotating those specific tickets gives more insight than running more prompt variants. They reveal where prompt design decisions actually matter.

---

## Recommendations

| Priority | Action |
|---|---|
| High | Expand dataset to 500+ tickets for statistically meaningful results |
| High | Run each prompt ≥5 times; report mean ± std |
| Medium | Add a `--repeat N` flag to `eval.py` for multi-run averaging |
| Medium | Update `baseline.py` to require a minimum margin (e.g. +4%) to pass |
| Low | Use `temperature=0` for fully deterministic evals, then re-introduce temperature to measure sensitivity |
| Low | Audit the borderline tickets that flip — they reveal category definition gaps |
