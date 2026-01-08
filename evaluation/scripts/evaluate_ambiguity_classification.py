#!/usr/bin/env python3
"""
Evaluation script for binary ambiguity detection.

This script evaluates the binary classification performance of the ambiguity
detection model on preprocessed datasets. It classifies queries as either
clear (label 0) or ambiguous (label 1).

Features:
- Binary classification: clear -> 0, ambiguous -> 1
- Classification report with precision, recall, F1-score
- Average inference time measurement
- Support for multiple datasets (ClariQ, AmbigNQ)
- Multithreaded processing for faster evaluation
- Progress tracking with tqdm

Usage:
    # Evaluate on ClariQ dataset
    python evaluate_ambiguity_classification.py --dataset clariq

    # Evaluate on AmbigNQ dataset
    python evaluate_ambiguity_classification.py --dataset ambignq

    # Evaluate on all datasets
    python evaluate_ambiguity_classification.py --dataset all

    # Custom configuration
    python evaluate_ambiguity_classification.py --dataset clariq --batch-size 64 --max-workers 16
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.clari_gen.clients import SmallModelClient
from core.clari_gen.prompts import BinaryDetectionPrompt
from core.clari_gen.utils.logger import setup_logger

logger = setup_logger(__name__)


class ProgressCounter:
    """Thread-safe counter for tracking processing progress."""

    def __init__(self):
        self.lock = Lock()
        self.count = 0

    def increment(self):
        with self.lock:
            self.count += 1
            return self.count


def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Load an ambiguity dataset from TSV file.

    Args:
        data_path: Path to the TSV file containing queries and labels

    Returns:
        DataFrame with 'initial_request' and 'binary_label' columns
    """
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path, sep="\t")

    # Validate required columns
    required_cols = ["initial_request", "binary_label"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    # Convert labels to int
    df["binary_label"] = df["binary_label"].astype(int)

    logger.info(f"Loaded {len(df)} queries")
    logger.info(f"Label distribution: {df['binary_label'].value_counts().to_dict()}")

    return df


def initialize_client() -> SmallModelClient:
    """
    Initialize the small model client for binary ambiguity detection.

    Returns:
        Configured SmallModelClient instance
    """
    logger.info("Initializing small model client for binary detection")

    try:
        small_client = SmallModelClient()

        # Test connection
        logger.info("Testing model server connection...")
        if not small_client.test_connection():
            raise ConnectionError(
                "Failed to connect to small model server (port 8368). "
                "Please ensure server is running with: cd server && ./serve_models.sh"
            )

        logger.info("✓ Model server connected successfully")
        return small_client

    except Exception as e:
        logger.error(f"Failed to initialize client: {e}")
        raise


def classify_single_query(
    small_client: SmallModelClient,
    query: str,
    query_idx: int,
    max_retries: int = 3,
    strategy: str = "few_shot",
) -> Tuple[int, str, str, float]:
    """
    Classify a single query as clear (0) or ambiguous (1) with retry mechanism.

    Args:
        small_client: The small model client for binary detection
        query: Query string to classify
        query_idx: Index of the query for ordering
        max_retries: Maximum number of retry attempts for parsing errors (default: 3)
        strategy: Prompting strategy - "zero_shot" or "few_shot" (default: "few_shot")

    Returns:
        Tuple of (query_idx, predicted_label, detection_result, error_msg, inference_time)
        - predicted_label: 0 if clear, 1 if ambiguous
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            start_time = time.time()

            # Create binary detection messages
            messages = BinaryDetectionPrompt.create_messages(query, strategy)

            # Call binary detection with structured output
            response = small_client.detect_binary_ambiguity(
                messages,
                response_format=BinaryDetectionPrompt.get_response_schema(),
            )

            # Parse structured response
            data = BinaryDetectionPrompt.parse_response(response)
            is_ambiguous = data["is_ambiguous"]

            inference_time = time.time() - start_time

            # Binary classification: False -> 0 (clear), True -> 1 (ambiguous)
            predicted_label = 1 if is_ambiguous else 0

            # Detection result as string
            detection_result = "AMBIGUOUS" if is_ambiguous else "CLEAR"

            return (query_idx, predicted_label, detection_result, "", inference_time)

        except KeyboardInterrupt:
            # Re-raise keyboard interrupt to allow clean shutdown
            raise
        except ValueError as e:
            # Parsing error - retry with exponential backoff
            last_error = e
            error_msg = str(e)
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed for query {query_idx} "
                f"'{query[:50]}...': {error_msg[:200]}"
            )

            # If this was the last attempt, log the full error and return default
            if attempt == max_retries - 1:
                logger.error(
                    f"All {max_retries} attempts failed for query {query_idx}. "
                    f"Last error: {error_msg[:300]}"
                )
                # Default to clear (label 0) on error after all retries
                return (query_idx, 0, "ERROR", error_msg, 0.0)

            # Wait before retrying (exponential backoff: 0.5s, 1s, 2s)
            time.sleep(0.5 * (2**attempt))

        except Exception as e:
            # Non-parsing errors fail immediately
            error_msg = str(e)
            logger.error(
                f"Unexpected error processing query {query_idx} '{query[:50]}...': {error_msg[:200]}"
            )
            # Default to clear (label 0) on error
            return (query_idx, 0, "ERROR", error_msg, 0.0)

    # Should never reach here, but just in case
    return (query_idx, 0, "ERROR", str(last_error), 0.0)


def process_batch_multithreaded(
    small_client: SmallModelClient,
    queries: List[str],
    batch_idx: int,
    total_batches: int,
    max_workers: int = 8,
    max_retries: int = 3,
    strategy: str = "few_shot",
) -> List[Tuple[int, str, str, float]]:
    """
    Process a batch of queries using multithreading.

    Args:
        small_client: The small model client for binary detection
        queries: List of query strings to process
        batch_idx: Current batch index (for logging)
        total_batches: Total number of batches (for logging)
        max_workers: Maximum number of concurrent threads
        max_retries: Maximum number of retry attempts for parsing errors (default: 3)
        strategy: Prompting strategy - "zero_shot" or "few_shot" (default: "few_shot")

    Returns:
        List of tuples: (predicted_label, detection_result, error_msg, inference_time)
    """
    results = [None] * len(queries)
    counter = ProgressCounter()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all queries for processing
            futures = {
                executor.submit(
                    classify_single_query,
                    small_client,
                    query,
                    idx,
                    max_retries,
                    strategy,
                ): idx
                for idx, query in enumerate(queries)
            }

            # Process results as they complete with progress bar
            with tqdm(
                total=len(queries),
                desc=f"Batch {batch_idx}/{total_batches}",
                leave=False,
            ) as pbar:
                for future in as_completed(futures):
                    try:
                        (
                            query_idx,
                            predicted_label,
                            detection_result,
                            error_msg,
                            inference_time,
                        ) = future.result()
                        results[query_idx] = (
                            predicted_label,
                            detection_result,
                            error_msg,
                            inference_time,
                        )
                        counter.increment()
                        pbar.update(1)
                    except KeyboardInterrupt:
                        logger.info(
                            "\nKeyboardInterrupt received, shutting down gracefully..."
                        )
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    except Exception as e:
                        query_idx = futures[future]
                        error_msg = str(e)
                        logger.error(
                            f"Thread execution error for query {query_idx}: {error_msg[:200]}"
                        )
                        results[query_idx] = (0, "ERROR", error_msg, 0.0)
                        pbar.update(1)

    except KeyboardInterrupt:
        logger.warning("\nBatch processing interrupted by user")
        raise

    # Fill any None results with error placeholders
    for idx, result in enumerate(results):
        if result is None:
            results[idx] = (0, "ERROR", "Query not processed", 0.0)

    return results


def evaluate_classification(
    data_path: str,
    batch_size: int = 32,
    output_path: str = None,
    max_workers: int = 8,
    max_retries: int = 3,
    strategy: str = "few_shot",
) -> Dict:
    """
    Evaluate the binary detection performance on the dataset.

    Args:
        data_path: Path to the dataset TSV file
        batch_size: Number of queries to process in each batch
        output_path: Optional path to save detailed results as TSV
        max_workers: Maximum number of concurrent threads (default: 8)
        max_retries: Maximum number of retry attempts for parsing errors (default: 3)
        strategy: Prompting strategy - "zero_shot" or "few_shot" (default: "few_shot")

    Returns:
        Dictionary containing evaluation metrics and results
    """
    # Load dataset
    df = load_dataset(data_path)

    # Initialize client
    client = initialize_client()

    # Prepare for batch processing
    queries = df["initial_request"].tolist()
    labels = df["binary_label"].tolist()

    total_queries = len(queries)
    total_batches = (total_queries + batch_size - 1) // batch_size

    logger.info(
        f"Processing {total_queries} queries in {total_batches} batches of size {batch_size}"
    )
    logger.info(f"Using multithreading with max_workers={max_workers}")
    logger.info(f"Max retries per query: {max_retries}")
    logger.info(f"Prompting strategy: {strategy}")

    # Process queries in batches
    all_predictions = []
    all_detection_results = []
    all_errors = []
    all_inference_times = []

    # Track total processing time
    total_processing_start = time.time()

    try:
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_queries)
            batch_queries = queries[start_idx:end_idx]

            batch_results = process_batch_multithreaded(
                client,
                batch_queries,
                batch_idx + 1,
                total_batches,
                max_workers,
                max_retries,
                strategy,
            )

            for (
                predicted_label,
                detection_result,
                error_msg,
                inference_time,
            ) in batch_results:
                all_predictions.append(predicted_label)
                all_detection_results.append(detection_result)
                all_errors.append(error_msg)
                all_inference_times.append(inference_time)

    except KeyboardInterrupt:
        logger.warning(
            f"\n\nEvaluation interrupted! Processed {len(all_predictions)}/{total_queries} queries"
        )
        logger.info("Generating partial results...")

        # Adjust labels to match the number of processed queries
        labels = labels[: len(all_predictions)]
        queries = queries[: len(all_predictions)]

        if len(all_predictions) == 0:
            logger.error("No queries were processed before interruption")
            raise

    # Calculate total processing time
    total_processing_time = time.time() - total_processing_start

    # Generate classification report
    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 60)

    # String report for display
    report_str = classification_report(
        labels, all_predictions, target_names=["Clear (0)", "Ambiguous (1)"], digits=4
    )
    print(report_str)
    logger.info("\n" + report_str)

    # Dict report for storage
    report_dict = classification_report(
        labels,
        all_predictions,
        target_names=["Clear (0)", "Ambiguous (1)"],
        output_dict=True,
        digits=4,
    )

    # Generate confusion matrix
    cm = confusion_matrix(labels, all_predictions)
    logger.info("\n" + "=" * 60)
    logger.info("CONFUSION MATRIX")
    logger.info("=" * 60)
    logger.info(f"\n{cm}")
    logger.info(f"\nTN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print(f"                Predicted Clear  Predicted Ambiguous")
    print(f"Actual Clear          {cm[0,0]:<10}     {cm[0,1]:<10}")
    print(f"Actual Ambiguous      {cm[1,0]:<10}     {cm[1,1]:<10}")
    print(f"\nTrue Negatives (TN): {cm[0,0]}")
    print(f"False Positives (FP): {cm[0,1]}")
    print(f"False Negatives (FN): {cm[1,0]}")
    print(f"True Positives (TP): {cm[1,1]}")

    # Calculate accuracy
    accuracy = accuracy_score(labels, all_predictions)

    # Calculate weighted metrics explicitly (accounts for class imbalance)
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(labels, all_predictions, average="weighted")
    )

    # Calculate inference time statistics
    valid_times = [t for t in all_inference_times if t > 0]
    avg_inference_time = (
        total_processing_time / len(all_predictions) if len(all_predictions) > 0 else 0
    )

    # Print inference time
    print("\n" + "=" * 60)
    print("INFERENCE TIME")
    print("=" * 60)
    print(f"Average inference time per query: {avg_inference_time:.4f} seconds")
    print(f"Total processing time: {total_processing_time:.2f} seconds")

    # Compile results
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_path": data_path,
        "total_queries": total_queries,
        "processed_queries": len(all_predictions),
        "batch_size": batch_size,
        "max_workers": max_workers,
        "classification_report": report_dict,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "avg_inference_time_seconds": avg_inference_time,
        "total_processing_time_seconds": total_processing_time,
        "error_count": sum(1 for e in all_errors if e),
    }

    # Save detailed results if output path provided
    if output_path:
        logger.info(f"\nSaving detailed results to {output_path}")

        results_df = pd.DataFrame(
            {
                "initial_request": queries,
                "ground_truth": labels,
                "predicted": all_predictions,
                "detection_result": all_detection_results,
                "error": all_errors,
                "correct": [
                    1 if g == p else 0 for g, p in zip(labels, all_predictions)
                ],
            }
        )

        results_df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"✓ Detailed results saved")

        # Also save metrics as JSON
        metrics_path = output_path.replace(".tsv", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"✓ Metrics saved to {metrics_path}")

    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Evaluate ambiguity classification on preprocessed datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on ClariQ dataset with default zero-shot CoT strategy
  python evaluate_ambiguity_classification.py --dataset clariq
  
  # Evaluate with different prompting strategies
  python evaluate_ambiguity_classification.py --dataset clariq --strategy zero_shot
  python evaluate_ambiguity_classification.py --dataset clariq --strategy few_shot
  python evaluate_ambiguity_classification.py --dataset clariq --strategy zero_shot_cot
  python evaluate_ambiguity_classification.py --dataset clariq --strategy few_shot_cot
  
  # Evaluate on AmbigNQ dataset
  python evaluate_ambiguity_classification.py --dataset ambignq
  
  # Evaluate on all datasets
  python evaluate_ambiguity_classification.py --dataset all
  
  # Use custom data path
  python evaluate_ambiguity_classification.py --data-path path/to/data.tsv
  
  # Adjust performance settings with few-shot CoT
  python evaluate_ambiguity_classification.py --batch-size 64 --max-workers 16 --strategy few_shot_cot
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["clariq", "ambignq", "all"],
        default="clariq",
        help="Dataset to evaluate on: 'clariq', 'ambignq', or 'all' (default: clariq)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the dataset TSV file (overrides --dataset if provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save detailed results as TSV (default: eval/results_DATASET_TIMESTAMP.tsv)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of queries to process in each batch (default: 32)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of concurrent threads (default: 8, recommended: 8-16)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts for queries that fail to parse (default: 3)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["zero_shot", "few_shot"],
        default="few_shot",
        help="Prompting strategy: 'zero_shot' (no examples) or 'few_shot' (with examples) (default: few_shot)",
    )

    args = parser.parse_args()

    # Determine which datasets to evaluate
    project_root = Path(__file__).resolve().parent.parent
    datasets_to_eval = []

    if args.data_path:
        # Custom data path
        dataset_name = (
            Path(args.data_path)
            .stem.replace("_preprocessed", "")
        )
        datasets_to_eval.append((dataset_name, args.data_path))
    elif args.dataset == "all":
        # Evaluate all datasets
        datasets_to_eval.append(
            ("clariq", str(project_root / "data" / "clariq_preprocessed.tsv"))
        )
        datasets_to_eval.append(
            ("ambignq", str(project_root / "data" / "ambignq_preprocessed.tsv"))
        )
    else:
        # Single dataset
        dataset_files = {
            "clariq": project_root / "data" / "clariq_preprocessed.tsv",
            "ambignq": project_root / "data" / "ambignq_preprocessed.tsv",
        }
        datasets_to_eval.append((args.dataset, str(dataset_files[args.dataset])))

    # Evaluate each dataset
    all_results = {}

    for dataset_name, data_path in datasets_to_eval:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"EVALUATING DATASET: {dataset_name.upper()}")
        logger.info(f"{'=' * 80}")
        logger.info(f"Data path: {data_path}")

        # Set output path if not provided
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results_{dataset_name}_{args.strategy}_{timestamp}.tsv"
        else:
            # If evaluating multiple datasets, append dataset name to output path
            if args.dataset == "all":
                output_base = Path(args.output).stem
                output_ext = Path(args.output).suffix
                output_path = (
                    f"{output_base}_{dataset_name}_{args.strategy}{output_ext}"
                )
            else:
                output_path = args.output

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        if output_dir != Path("."):
            output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Run evaluation
            results = evaluate_classification(
                data_path=data_path,
                batch_size=args.batch_size,
                output_path=output_path,
                max_workers=args.max_workers,
                max_retries=args.max_retries,
                strategy=args.strategy,
            )

            all_results[dataset_name] = results

            # Print summary
            print("\n" + "=" * 60)
            print(f"EVALUATION SUMMARY - {dataset_name.upper()}")
            print("=" * 60)
            print(f"Total Queries: {results['total_queries']}")
            print(f"Processed Queries: {results['processed_queries']}")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"Weighted F1: {results['weighted_f1']:.4f}")
            print(f"Weighted Precision: {results['weighted_precision']:.4f}")
            print(f"Weighted Recall: {results['weighted_recall']:.4f}")
            print(f"\nFor Ambiguous Class (1):")
            print(
                f"  Precision: {results['classification_report']['Ambiguous (1)']['precision']:.4f}"
            )
            print(
                f"  Recall: {results['classification_report']['Ambiguous (1)']['recall']:.4f}"
            )
            print(
                f"  F1-Score: {results['classification_report']['Ambiguous (1)']['f1-score']:.4f}"
            )

            print(f"\nInference Performance:")
            print(f"  Avg Time per Query: {results['avg_inference_time_seconds']:.4f}s")
            print(
                f"  Total Processing Time: {results['total_processing_time_seconds']:.2f}s"
            )

            if results["error_count"] > 0:
                print(
                    f"\nWarning: {results['error_count']} queries had errors during processing"
                )

            if results["processed_queries"] < results["total_queries"]:
                print(
                    f"\nNote: Evaluation was incomplete - only {results['processed_queries']}/{results['total_queries']} queries processed"
                )

            print(f"\n✓ Evaluation completed successfully for {dataset_name}")

        except KeyboardInterrupt:
            logger.warning(f"\n\nEvaluation interrupted by user (Ctrl+C)")
            print(f"\nPartial results may have been saved to: {output_path}")
            if len(datasets_to_eval) > 1:
                print(f"\nSkipping remaining datasets...")
            break
        except Exception as e:
            logger.error(f"Evaluation failed for {dataset_name}: {e}")
            import traceback

            traceback.print_exc()
            if len(datasets_to_eval) > 1:
                print(f"\nContinuing to next dataset...")
                continue
            else:
                sys.exit(1)

    # Print combined summary if multiple datasets
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("COMBINED SUMMARY")
        print("=" * 80)
        for dataset_name, results in all_results.items():
            print(f"\n{dataset_name.upper()}:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Weighted F1: {results['weighted_f1']:.4f}")
            print(f"  Avg Inference Time: {results['avg_inference_time_seconds']:.4f}s")

    print("\n✓ All evaluations completed")


if __name__ == "__main__":
    main()
