#!/usr/bin/env python3
"""
Evaluation script for ambiguity detection pipeline.

This script evaluates the classification of ambiguity types to detect ambiguous queries
using ambiguity datasets (ClariQ or AmbigNQ). It processes queries in batches using
multithreading for parallel processing, compares predictions against ground truth
labels, and generates a classification report with weighted metrics.

Features:
- Multithreaded processing for faster evaluation
- Support for multiple datasets (ClariQ, AmbigNQ)
- Batch processing with configurable batch size
- Progress tracking with tqdm
- Thread-safe result collection
- TSV and JSON output formats

Usage:
    # Evaluate on ClariQ dataset
    # python evaluate_ambiguity_detection.py --dataset clariq

    # Evaluate on AmbigNQ dataset
    python evaluate_ambiguity_detection.py --dataset ambignq

    # Custom configuration
    python evaluate_ambiguity_detection.py --dataset clariq --batch-size 64 --max-workers 16
"""

import argparse
import sys
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from clari_gen.clients import SmallModelClient
from clari_gen.prompts import AmbiguityClassificationPrompt
from clari_gen.models import AmbiguityType
from clari_gen.utils.logger import setup_logger

logger = setup_logger(__name__)


# Thread-safe counter for progress tracking
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


def initialize_pipeline() -> SmallModelClient:
    """
    Initialize the small model client for ambiguity classification.
    
    For evaluation purposes, we only need classification (not clarification generation),
    so we bypass the full pipeline and use only the small model directly.

    Returns:
        Configured SmallModelClient instance
    """
    logger.info("Initializing small model client for classification")

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


def process_single_query(
    small_client: SmallModelClient, query: str, query_idx: int
) -> Tuple[int, bool, str, str, float]:
    """
    Process a single query through classification only (optimized for evaluation).
    
    This bypasses the full pipeline and only performs ambiguity classification,
    skipping clarification generation for faster evaluation.

    Args:
        small_client: The small model client for classification
        query: Query string to process
        query_idx: Index of the query for ordering

    Returns:
        Tuple of (query_idx, is_ambiguous, ambiguity_types, error_msg, inference_time)
    """
    try:
        start_time = time.time()
        
        # Create classification messages
        messages = AmbiguityClassificationPrompt.create_messages(query)
        
        # Call classification WITH structured output (using Outlines backend)
        # This enforces JSON schema compliance but is slower than unstructured generation
        response = small_client.classify_ambiguity(
            messages,
            response_format=AmbiguityClassificationPrompt.get_response_schema(),
        )
        
        # Parse structured response
        ambiguity_types_strs, reasoning = AmbiguityClassificationPrompt.parse_response(response)
        
        inference_time = time.time() - start_time

        # Determine if ambiguous (NONE means not ambiguous)
        is_ambiguous = "NONE" not in ambiguity_types_strs
        
        # Get ambiguity types as string
        ambiguity_types = ", ".join(ambiguity_types_strs)

        return (query_idx, is_ambiguous, ambiguity_types, "", inference_time)

    except KeyboardInterrupt:
        # Re-raise keyboard interrupt to allow clean shutdown
        raise
    except Exception as e:
        # Log full error for debugging but continue processing
        error_msg = str(e)
        logger.warning(
            f"Error processing query {query_idx} '{query[:50]}...': {error_msg[:200]}"
        )
        # Default to not ambiguous on error (label 0)
        return (query_idx, False, "ERROR", error_msg, 0.0)


def process_batch_multithreaded(
    small_client: SmallModelClient,
    queries: List[str],
    batch_idx: int,
    total_batches: int,
    max_workers: int = 8,
) -> List[Tuple[bool, str, str, float]]:
    """
    Process a batch of queries using multithreading (classification only).

    Args:
        small_client: The small model client for classification
        queries: List of query strings to process
        batch_idx: Current batch index (for logging)
        total_batches: Total number of batches (for logging)
        max_workers: Maximum number of concurrent threads

    Returns:
        List of tuples: (is_ambiguous, ambiguity_types, error_msg, inference_time)
    """
    results = [None] * len(queries)
    counter = ProgressCounter()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all queries for processing
            futures = {
                executor.submit(process_single_query, small_client, query, idx): idx
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
                            is_ambiguous,
                            ambiguity_types,
                            error_msg,
                            inference_time,
                        ) = future.result()
                        results[query_idx] = (
                            is_ambiguous,
                            ambiguity_types,
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
                        results[query_idx] = (False, "ERROR", error_msg, 0.0)
                        pbar.update(1)

    except KeyboardInterrupt:
        logger.warning("\nBatch processing interrupted by user")
        raise

    # Fill any None results with error placeholders
    for idx, result in enumerate(results):
        if result is None:
            results[idx] = (False, "ERROR", "Query not processed", 0.0)

    return results


def evaluate_pipeline(
    data_path: str,
    batch_size: int = 32,
    output_path: str = None,
    max_workers: int = 8,
) -> Dict:
    """
    Evaluate the pipeline on the dataset and generate metrics.

    Args:
        data_path: Path to the dataset TSV file
        batch_size: Number of queries to process in each batch
        output_path: Optional path to save detailed results as TSV
        max_workers: Maximum number of concurrent threads (default: 8)

    Returns:
        Dictionary containing evaluation metrics and results
    """
    # Load dataset
    df = load_dataset(data_path)

    # Initialize pipeline
    pipeline = initialize_pipeline()

    # Prepare for batch processing
    queries = df["initial_request"].tolist()
    labels = df["binary_label"].tolist()

    total_queries = len(queries)
    total_batches = (total_queries + batch_size - 1) // batch_size

    logger.info(
        f"Processing {total_queries} queries in {total_batches} batches of size {batch_size}"
    )
    logger.info(f"Using multithreading with max_workers={max_workers}")

    # Process queries in batches
    all_predictions = []
    all_ambiguity_types = []
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
                pipeline, batch_queries, batch_idx + 1, total_batches, max_workers
            )

            for (
                is_ambiguous,
                ambiguity_types,
                error_msg,
                inference_time,
            ) in batch_results:
                all_predictions.append(1 if is_ambiguous else 0)
                all_ambiguity_types.append(ambiguity_types)
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

    # Calculate additional metrics

    # Calculate weighted metrics using precision_recall_fscore_support
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, all_predictions, average='weighted', zero_division=0
    )

    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, all_predictions, average=None, labels=[0, 1], zero_division=0
    )
    
    # Class 1 is Ambiguous (index 1)
    precision_ambiguous = precision_per_class[1]
    recall_ambiguous = recall_per_class[1]
    f1_ambiguous = f1_per_class[1]

    # Calculate accuracy
    accuracy = accuracy_score(labels, all_predictions)

    # Calculate total processing time
    total_processing_time = time.time() - total_processing_start

    # Compile results
    # Calculate inference time statistics
    valid_times = [t for t in all_inference_times if t > 0]
    # Use amortized time for average to reflect system throughput
    avg_inference_time = total_processing_time / len(all_predictions) if len(all_predictions) > 0 else 0

    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_path": data_path,
        "total_queries": total_queries,
        "processed_queries": len(all_predictions),
        "batch_size": batch_size,
        "classification_report": report_dict,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "summary_metrics": {
            "accuracy": accuracy,
            "precision_ambiguous": precision_ambiguous,
            "recall_ambiguous": recall_ambiguous,
            "f1_ambiguous": f1_ambiguous,
            "weighted_avg_f1": f1_weighted,
            "weighted_avg_precision": precision_weighted,
            "weighted_avg_recall": recall_weighted,
        },
        "inference_time_stats": {
            "avg_time_seconds": avg_inference_time
        },
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
                "ambiguity_types": all_ambiguity_types,
                "inference_time_seconds": all_inference_times,
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
        description="Evaluate ambiguity detection pipeline on ambiguity datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on ClariQ dataset (default)
  python evaluate_ambiguity_detection.py --dataset clariq
  
  # Evaluate on AmbigNQ dataset
  python evaluate_ambiguity_detection.py --dataset ambignq
  
  # Use custom data path
  python evaluate_ambiguity_detection.py --data-path path/to/data.tsv
  
  # Adjust performance settings
  python evaluate_ambiguity_detection.py --batch-size 64 --max-workers 16
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["clariq", "ambignq"],
        default="clariq",
        help="Dataset to evaluate on: 'clariq' or 'ambignq' (default: clariq)",
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

    args = parser.parse_args()

    # Determine data path based on dataset argument or custom path
    if args.data_path:
        data_path = args.data_path
        dataset_name = Path(args.data_path).stem.replace("_preprocessed", "")
    else:
        dataset_name = args.dataset
        project_root = Path(__file__).resolve().parent.parent
        dataset_files = {
            "clariq": project_root / "data" / "clariq_preprocessed.tsv",
            "ambignq": project_root / "data" / "ambignq_preprocessed.tsv",
        }
        data_path = str(dataset_files[args.dataset])

    logger.info(f"Using dataset: {dataset_name}")
    logger.info(f"Data path: {data_path}")

    # Set default output path if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results_{dataset_name}_{timestamp}.tsv"

    # Ensure output directory exists
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run evaluation
        results = evaluate_pipeline(
            data_path=data_path,
            batch_size=args.batch_size,
            output_path=args.output,
            max_workers=args.max_workers,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Dataset: {dataset_name}")
        print(f"Total Queries: {results['total_queries']}")
        print(f"Processed Queries: {results['processed_queries']}")
        print(f"Accuracy: {results['summary_metrics']['accuracy']:.4f}")
        print(f"Weighted F1: {results['summary_metrics']['weighted_avg_f1']:.4f}")
        print(
            f"Weighted Precision: {results['summary_metrics']['weighted_avg_precision']:.4f}"
        )
        print(
            f"Weighted Recall: {results['summary_metrics']['weighted_avg_recall']:.4f}"
        )
        print(f"\nFor Ambiguous Class (1):")
        print(f"  Precision: {results['summary_metrics']['precision_ambiguous']:.4f}")
        print(f"  Recall: {results['summary_metrics']['recall_ambiguous']:.4f}")
        print(f"  F1-Score: {results['summary_metrics']['f1_ambiguous']:.4f}")

        print(f"\nInference Performance:")
        print(
            f"  Avg Time per Query: {results['inference_time_stats']['avg_time_seconds']:.3f}s (amortized)"
        )

        if results["error_count"] > 0:
            print(
                f"\nWarning: {results['error_count']} queries had errors during processing"
            )

        if results["processed_queries"] < results["total_queries"]:
            print(
                f"\nNote: Evaluation was incomplete - only {results['processed_queries']}/{results['total_queries']} queries processed"
            )

        print("\n✓ Evaluation completed successfully")

    except KeyboardInterrupt:
        logger.warning("\n\nEvaluation interrupted by user (Ctrl+C)")
        print("\nPartial results may have been saved to:", args.output)
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
