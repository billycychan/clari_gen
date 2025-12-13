"""
Evaluate clarification generation module using BERTScore.
"""

import argparse
import ast
import logging
import os
import sys
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm

# Add the project root to the path so we can import clari_gen
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from clari_gen.clients.large_model_client import LargeModelClient
from clari_gen.prompts.clarification_generation.at_cot import ClarificationATCoTPrompt
from clari_gen.prompts.clarification_generation.at_standard import ClarificationATStandardPrompt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
import transformers
transformers.logging.set_verbosity_error()

# Try to import bert_score
try:
    from bert_score import score as bert_score_func
except ImportError:
    logger.warning("bert_score not installed. BERTScore calculation will be skipped or fail.")
    bert_score_func = None


def load_data(filepath: str) -> pd.DataFrame:
    """Load the dataset and group by query."""
    df = pd.read_csv(filepath, sep="\t")
    # Group by query and aggregate questions into a list
    grouped_df = df.groupby("query")["question"].apply(list).reset_index()
    return grouped_df


def generate_candidates(
    client: LargeModelClient,
    query: str,
    prompt_cls: Any,
    num_candidates: int,
    max_workers: int = 20,
) -> List[str]:
    """Generate candidates in parallel using the specified prompt class and client."""
    messages = prompt_cls.create_messages(query)
    
    import concurrent.futures
    
    candidates = [None] * num_candidates
    
    def generate_single(index):
        try:
            response_text = client.generate(
                messages=messages,
                temperature=0.7, 
                max_tokens=512, # Increased from 256
                response_format=prompt_cls.get_response_schema()
            )
            parsed = prompt_cls.parse_response(response_text)
            return index, parsed["clarifying_question"]
        except Exception as e:
            logger.error(f"Error generating candidate {index} for query '{query}': {e}")
            return index, ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_single, i) for i in range(num_candidates)]
        
        for future in concurrent.futures.as_completed(futures):
            idx, question = future.result()
            candidates[idx] = question
            
    return candidates


def evaluate_query(
    candidates: List[str], references: List[str]
) -> float:
    """
    Calculate the max BERTScore from the NxN matrix.
    candidates: List of generated questions (length N)
    references: List of reference questions (length N)
    
    Returns the single best score for this query.
    """
    if not candidates or not references:
        return 0.0
    
    if bert_score_func is None:
        # If we are here, it means the import failed initially.
        # But if the user ran in a different environment where it is installed,
        # we might want to try importing again or fail.
        # However, the top-level import is what matters.
        logger.error("bert_score library not found. Returning 0.0.")
        return 0.0

    try:
        # Filter out empty candidates
        valid_candidates = [c for c in candidates if c]
        if not valid_candidates:
            return 0.0

        # Create pairs: every valid candidate against every reference
        all_cands = []
        all_refs = []
        
        for c in valid_candidates:
            for r in references:
                if not r: continue
                all_cands.append(c)
                all_refs.append(r)
        
        if not all_cands:
            return 0.0

        # Calculate BERTScore
        # Suppress warnings and progress bars
        P, R, F1 = bert_score_func(all_cands, all_refs, lang="en", verbose=False, model_type="roberta-large")
        
        max_f1 = F1.max().item()
        return max_f1

    except Exception as e:
        logger.error(f"Error calculating BERTScore: {e}")
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate clarification generation.")
    parser.add_argument("--data_path", default="data/clariq_cq.tsv", help="Path to data file")
    parser.add_argument("--output_dir", default="eval_results", help="Directory to save results")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of queries to evaluate (for testing)")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers for generation")
    parser.add_argument(
        "--prompt_type", 
        choices=["all", "standard", "cot"], 
        default="all", 
        help="Prompt method to evaluate (standard=AT_Standard, cot=AT_CoT)"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading data from {args.data_path}")
    grouped_df = load_data(args.data_path)
    
    if args.num_examples:
        grouped_df = grouped_df.head(args.num_examples)
        logger.info(f"Running on first {args.num_examples} examples")

    # Initialize client
    client = LargeModelClient(base_url="http://localhost:8369/v1") 

    results = []

    all_methods = [
        ("AT_Standard", ClarificationATStandardPrompt),
        ("AT_CoT", ClarificationATCoTPrompt),
    ]
    
    methods = []
    if args.prompt_type == "all":
        methods = all_methods
    elif args.prompt_type == "standard":
        methods = [m for m in all_methods if m[0] == "AT_Standard"]
    elif args.prompt_type == "cot":
        methods = [m for m in all_methods if m[0] == "AT_CoT"]

    for index, row in tqdm(grouped_df.iterrows(), total=len(grouped_df), desc="Evaluating queries"):
        query = row["query"]
        references = row["question"]
        num_refs = len(references)
        
        logger.info(f"Processing query: '{query}' ({num_refs} references)")

        query_result = {
            "query": query,
            "num_references": num_refs,
        }
        
        for method_name, prompt_cls in methods:
            try:
                # Generate N candidates in parallel
                candidates = generate_candidates(client, query, prompt_cls, num_refs, max_workers=args.workers)
                
                # Calculate score
                score = evaluate_query(candidates, references)
                
                query_result[f"{method_name}_score"] = score
                query_result[f"{method_name}_candidates"] = candidates
                logger.info(f"{method_name} Score: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {method_name} for query '{query}': {e}")
                query_result[f"{method_name}_score"] = 0.0
                query_result[f"{method_name}_candidates"] = []

        results.append(query_result)

    # Save results
    results_df = pd.DataFrame(results)
    
    data_filename = os.path.splitext(os.path.basename(args.data_path))[0]
    output_filename = f"{data_filename}_clarification_evaluation_results.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    
    results_df.to_csv(output_path, index=False)
    logger.info(f"Detailed results saved to {output_path}")

    # Calculate and print aggregate metrics
    print("\nAggregate Results:")
    for method_name, _ in methods:
        mean_score = results_df[f"{method_name}_score"].mean()
        print(f"{method_name} Mean BERTScore: {mean_score:.4f}")


if __name__ == "__main__":
    main()
