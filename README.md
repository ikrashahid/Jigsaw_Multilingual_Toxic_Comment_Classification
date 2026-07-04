# Responsible & Explainable AI 

This repository contains a full starter implementation for the five required assignment parts.
For the actual executed run, the work is consolidated in `actual_run.ipynb` (single notebook covering multiple phases).

- `part1.ipynb` - baseline DistilBERT training + threshold analysis
- `part2.ipynb` - subgroup fairness audit
- `part3.ipynb` - adversarial evasion + poisoning
- `part4.ipynb` - mitigation setup and comparison scaffold
- `part5.ipynb` - production-style guardrail pipeline demo
- `actual_run.ipynb` - consolidated executed notebook (Phase 1 + Phase 2, with outputs)
- `pipeline.py` - `ModerationPipeline` class with 3 layers
- `assignment_workflow.py` - shared reusable functions
- `requirements.txt` - pinned dependencies

## Python / Hardware

- Python: `3.10+`
- Recommended runtime: Google Colab GPU (T4) or local CUDA GPU

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data

Place Kaggle files in repository root:

- `jigsaw-unintended-bias-train.csv` (required)
- `validation.csv` (optional)

The dataset itself should not be committed.

## Execution Format

- Primary executed notebook: `actual_run.ipynb`
- This notebook contains the code run for the assignment phases in one place.
- The `part1.ipynb` to `part5.ipynb` files are kept for assignment structure/reference.

## Run Order

1. Open and run `actual_run.ipynb` end-to-end on Colab GPU.
2. Verify all required outputs (metrics, plots, and tables) are visible.
3. Save the executed notebook and push to GitHub.
4. Keep `part1.ipynb` to `part5.ipynb` in the repository for mapping to assignment parts.

## Important Notes

- Submit notebooks **with outputs rendered**.
- Keep incremental commit history (avoid single-commit submission).
- Keep large artifacts out of Git (`.gitignore` is already configured).

