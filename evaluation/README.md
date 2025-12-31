# ClariGen Evaluation Scripts

[← Back to Root](../README.md)

This directory contains scripts and datasets for evaluating the performance of the ClariGen ambiguity detection and clarification system.

## Setup

Ensure the core library is installed:
```bash
cd ../core
pip install -e .
```

Install evaluation dependencies:
```bash
pip install pandas scikit-learn tabulate
```

## Available Scripts

### 1. Ambiguity Classification Evaluation
Evaluates binary ambiguity detection performance.

```bash
python scripts/evaluate_ambiguity_classification.py \
  --dataset data/ambignq_preprocessed.tsv \
  --prompt-type zero-shot \
  --num-workers 10 \
  --batch-size 50
```

Options:
- `--dataset` - Path to dataset TSV file
- `--prompt-type` - Prompt strategy: `zero-shot` or `few-shot`
- `--num-workers` - Number of parallel workers
- `--batch-size` - Batch size for processing
- `--max-samples` - Limit number of samples (for testing)

### 2. Clarification Generation Evaluation
Evaluates the quality of generated clarifying questions.

```bash
python scripts/evaluate_clarification_generation.py \
  --dataset data/ambignq_preprocessed.tsv \
  --prompt-method at_cot \
  --num-workers 5
```

Options:
- `--dataset` - Path to dataset TSV file
- `--prompt-method` - Method: `at_standard`, `at_cot`, or `vanilla`
- `--num-workers` - Number of parallel workers
- `--max-samples` - Limit number of samples

### 3. Stability Analysis
Evaluates consistency across multiple runs.

```bash
python scripts/evaluate_stability.py \
  --input-file data/real-queries.tsv \
  --num-iterations 10 \
  --prompt-type zero-shot
```

Options:
- `--input-file` - Path to input queries TSV
- `--num-iterations` - Number of iterations per query
- `--prompt-type` - Prompt strategy: `zero-shot` or `few-shot`

### 4. Test Real Queries
Quick script to test ambiguity detection on custom queries.

```bash
python scripts/test_real_queries.py
```

Reads queries from `real-queries.tsv` and outputs results.

## Datasets

Evaluation datasets are located in `evaluation/data/`:
- `ambignq_preprocessed.tsv` - AmbigNQ dataset for evaluation
- `clariq_preprocessed.tsv` - ClariQ dataset
- `gpt_4_1_balanced_strict.tsv` - GPT-4.1 labeled dataset

## Results

Evaluation results are saved to `evaluation/results/`:
- Classification metrics (precision, recall, F1-score)
- Confusion matrices
- Timing information
- Generated clarifications

## Batch Evaluation

For comprehensive evaluation:

```bash
cd scripts
bash eval_binary_detection_datasets.sh
```

This will run evaluation on all datasets with both zero-shot and few-shot prompts.
