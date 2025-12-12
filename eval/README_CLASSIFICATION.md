# Ambiguity Classification Evaluation

This script evaluates the binary classification performance of the ambiguity classification model.

## Classification Logic

- **Label 0 (Clear)**: Query is classified as NONE (no ambiguity)
- **Label 1 (Ambiguous)**: Query is classified as any ambiguity type (LEXICAL, SEMANTIC, REFERENCE, UNFAMILIAR, CONTRADICTION)

## Usage

### Evaluate on ClariQ dataset
```bash
python evaluate_ambiguity_classification.py --dataset clariq
```

### Evaluate on AmbigNQ dataset
```bash
python evaluate_ambiguity_classification.py --dataset ambignq
```

### Evaluate on both datasets
```bash
python evaluate_ambiguity_classification.py --dataset both
```

### Custom configuration
```bash
python evaluate_ambiguity_classification.py --dataset clariq --batch-size 64 --max-workers 16
```

## Output

The script generates:

1. **Classification Report**: Precision, Recall, F1-score for each class
2. **Confusion Matrix**: TN, FP, FN, TP counts
3. **Inference Time**: Average time per query
4. **Detailed Results**: TSV file with per-query predictions
5. **Metrics JSON**: JSON file with all metrics

### Example Output

```
CLASSIFICATION REPORT
==============================================================
              precision    recall  f1-score   support

   Clear (0)     0.8500    0.9200    0.8836       100
Ambiguous (1)     0.9100    0.8300    0.8684       200

    accuracy                         0.8800       300
   macro avg     0.8800    0.8750    0.8760       300
weighted avg     0.8833    0.8800    0.8810       300

CONFUSION MATRIX
==============================================================
                Predicted Clear  Predicted Ambiguous
Actual Clear          92             8
Actual Ambiguous      34           166

INFERENCE TIME
==============================================================
Average inference time per query: 0.1234 seconds
Total processing time: 37.02 seconds
```

## Dataset Format

Both test datasets (`ambignq_preprocessed.tsv` and `clariq_preprocessed.tsv`) have the same structure:

| Column | Description |
|--------|-------------|
| `initial_request` | The query text |
| `binary_label` | Expected label (0=clear, 1=ambiguous) |

## Requirements

- vLLM server running on port 8368 with the small model (Llama-3.1-8B-Instruct)
- Run `cd server && ./serve_models.sh` to start the server
