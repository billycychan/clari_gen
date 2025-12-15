# Evaluation Summary — 2025-12-12

This file summarizes classification results for AmbigNQ and ClariQ (zero-shot vs few-shot).

**AmbigNQ**

| Shot | Total | Accuracy | Weighted F1 | Weighted Precision | Weighted Recall | Avg inference (s) | Confusion (tn / fp / fn / tp) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | 2002 | 0.51798 | 0.52061 | 0.54047 | 0.51798 | 0.017385 | 476 / 354 / 611 / 561 |
| Few-shot  | 2002 | 0.55445 | 0.51180 | 0.51948 | 0.55445 | 0.017396 | 171 / 659 / 233 / 939 |

**ClariQ**

| Shot | Total | Accuracy | Weighted F1 | Weighted Precision | Weighted Recall | Avg inference (s) | Confusion (tn / fp / fn / tp) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | 299 | 0.71572 | 0.75433 | 0.81282 | 0.71572 | 0.017579 | 15 / 22 / 63 / 199 |
| Few-shot  | 299 | 0.75585 | 0.77099 | 0.78801 | 0.75585 | 0.017599 | 7 / 30 / 43 / 219 |

**Key observations (concise)**

- Few-shot improves overall accuracy on both datasets (AmbigNQ: 0.518 → 0.554; ClariQ: 0.716 → 0.756).
- AmbigNQ shows a trade-off: few-shot raises recall for the Ambiguous class (higher TP) but increases false positives for Clear examples.
- ClariQ benefits in weighted F1 from few-shot prompting, with small change in inference latency (≈0.00002s).
