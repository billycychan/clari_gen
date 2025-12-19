
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import glob

from clari_gen.clients import SmallModelClient
from clari_gen.prompts import BinaryDetectionPrompt
from clari_gen.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

def classify_single_query(client, query, strategy="few_shot"):
    try:
        messages = BinaryDetectionPrompt.create_messages(query, strategy=strategy)
        response = client.detect_binary_ambiguity(
            messages,
            response_format=BinaryDetectionPrompt.get_response_schema(),
        )
        data = BinaryDetectionPrompt.parse_response(response)
        return data["is_ambiguous"]
    except Exception as e:
        return None

def main():
    # Configuration
    NUM_ITERATIONS = 100
    MAX_WORKERS = 16
    STRATEGIES = ["zero_shot", "few_shot"]
    INPUT_FILE = Path("real-queries.tsv")
    
    # Check input file first
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} does not exist.")
        return

    try:
        df = pd.read_csv(INPUT_FILE, sep="\t", header=None, names=["id", "query"])
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    queries = df["query"].tolist()
    ids = df["id"].tolist()
    print(f"Loaded {len(queries)} queries.")

    # Initialize client
    print("Initializing model client...")
    try:
        client = SmallModelClient()
        if not client.test_connection():
             print("Error: Could not connect to model server.")
             return
    except Exception as e:
        print(f"Error initializing client: {e}")
        return

    for STRATEGY in STRATEGIES:
        print(f"\n\nRunning stability test for {NUM_ITERATIONS} iterations per query...")
        print(f"Strategy: {STRATEGY}")
        
        # Dictionary to store results: query_id -> list of boolean results
        results_map = defaultdict(list)
        
        total_tasks = len(queries) * NUM_ITERATIONS
        
        print(f"Processing {total_tasks} total inference calls with {MAX_WORKERS} workers...")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Create all futures
            futures = []
            for _ in range(NUM_ITERATIONS):
                for q_id, query in zip(ids, queries):
                    future = executor.submit(classify_single_query, client, query, STRATEGY)
                    futures.append((future, q_id))
            
            # Process as they complete
            with tqdm(total=total_tasks, desc=f"Progress ({STRATEGY})") as pbar:
                for future, q_id in futures:
                    try:
                        is_ambiguous = future.result()
                        if is_ambiguous is not None:
                            results_map[q_id].append(is_ambiguous)
                    except Exception:
                        pass
                    pbar.update(1)

        duration = time.time() - start_time
        print(f"\nCompleted {STRATEGY} in {duration:.2f} seconds (Avg {duration/total_tasks:.4f}s per call)")

        # Generate Summary for this strategy
        summary_data = []

        print(f"\n--- Results for {STRATEGY} ---")
        print(f"{'ID':<5} | {'Ambiguous %':<12} | {'Clear %':<10} | {'Total Valid':<12} | {'Query'}")
        print("-" * 100)

        for q_id, query in zip(ids, queries):
            results = results_map[q_id]
            total_valid = len(results)
            
            if total_valid == 0:
                 summary_data.append({
                    "id": q_id,
                    "query": query,
                    "ambiguous_pct": 0,
                    "clear_pct": 0,
                    "total_runs": 0,
                    "strategy": STRATEGY
                })
                 continue
                
            ambiguous_count = sum(results)
            clear_count = total_valid - ambiguous_count
            
            amb_pct = (ambiguous_count / total_valid) * 100
            clear_pct = (clear_count / total_valid) * 100
            
            display_query = (query[:60] + '...') if len(query) > 60 else query
            
            print(f"{str(q_id):<5} | {amb_pct:<11.1f}% | {clear_pct:<9.1f}% | {total_valid:<12} | {display_query}")
            
            summary_data.append({
                "id": q_id,
                "query": query,
                "ambiguous_pct": amb_pct,
                "clear_pct": clear_pct,
                "total_runs": total_valid,
                "strategy": STRATEGY
            })

        # Save summary
        output_file = Path(f"stability_analysis_results_{STRATEGY}.tsv")
        pd.DataFrame(summary_data).to_csv(output_file, sep="\t", index=False)
        print(f"Detailed summary for {STRATEGY} saved to {output_file.absolute()}")

if __name__ == "__main__":
    main()
