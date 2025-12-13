# Evaluation: Clarification Generation

This folder contains the evaluation script for clarifying question generation using BERTScore.

Run the evaluation script:

```bash
python eval/evaluate_clarification_generation.py --dataset qulac --methods both --k 36
```

Outputs are written to `eval/results/` with timestamped filenames (per-method TSV and metrics JSON).

Dependencies:
- bert-score (add to `requirements.txt`)

Run unit tests (recommended):

```bash
pytest tests/test_evaluate_clarification_generation.py -q
```

