# Responsible & Explainable AI - Assignment 2

This repository contains a full starter implementation for the five required assignment parts:

- `part1.ipynb` - baseline DistilBERT training + threshold analysis
- `part2.ipynb` - subgroup fairness audit
- `part3.ipynb` - adversarial evasion + poisoning
- `part4.ipynb` - mitigation setup and comparison scaffold
- `part5.ipynb` - production-style guardrail pipeline demo
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

## Run Order

1. Run `part1.ipynb` end-to-end and keep outputs.
2. Run `part2.ipynb` using the saved baseline model.
3. Run `part3.ipynb` to evaluate attacks.
4. Complete + run `part4.ipynb` mitigation experiments.
5. Run `part5.ipynb` pipeline demonstration on 1000 samples.

## Important Notes

- Submit notebooks **with outputs rendered**.
- Keep incremental commit history (avoid single-commit submission).
- Keep large artifacts out of Git (`.gitignore` is already configured).

