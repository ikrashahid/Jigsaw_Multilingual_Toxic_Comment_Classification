# How This Assignment Flow Works

This file explains the project logic in plain language.

## End-to-End Design

1. **Shared utilities** are centralized in `assignment_workflow.py` so repeated code (loading, splitting, tokenizing, metrics, plotting) stays consistent across all notebook parts.
2. **Each part notebook** focuses on one rubric section from the PDF and imports shared helpers.
3. **Model handoff** is done via saved checkpoints (`saved_model/...`) so later parts do not retrain unnecessarily.
4. **Production layer** is in `pipeline.py` as required, with:
   - Layer 1: Regex pre-filter (auditable category-based blocklist)
   - Layer 2: Calibrated model decision
   - Layer 3: Human review queue for uncertain band

## Why This Is Efficient

- Avoids rewriting the same preprocessing/training logic in 5 notebooks.
- Makes debugging easier because all core logic is in one reusable module.
- Lets you iterate quickly on thresholds and fairness analysis.

## What You Still Need To Do

- Execute all notebooks on GPU and keep outputs visible.
- Finish the mitigation experiments in `part4.ipynb` (reweighing/oversampling loops and final table).
- Add your own markdown discussion answers for each “Key question” in the PDF.
- Capture final numbers/plots generated in your run environment.

