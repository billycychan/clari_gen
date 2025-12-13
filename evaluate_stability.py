
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

def classify_single_query(client, query, strategy="zero_shot"):
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
    STRATEGY = "zero_shot"
    INPUT_FILE = Path("real-queries.tsv")

    print(f"Running stability test for {NUM_ITERATIONS} iterations per query...")
    print(f"Strategy: {STRATEGY}")
    
    # Load queries
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

    # Dictionary to store results: query_id -> list of boolean results
    results_map = defaultdict(list)
    
    total_tasks = len(queries) * NUM_ITERATIONS
    
    print(f"\nProcessing {total_tasks} total inference calls with {MAX_WORKERS} workers...")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create all futures
        # We store (q_id, query) in the futures map to map results back
        futures = []
        for _ in range(NUM_ITERATIONS):
            for q_id, query in zip(ids, queries):
                future = executor.submit(classify_single_query, client, query, STRATEGY)
                futures.append((future, q_id))
        
        # Process as they complete
        with tqdm(total=total_tasks, desc="Progress") as pbar:
            for future, q_id in futures:
                try:
                    is_ambiguous = future.result()
                    if is_ambiguous is not None:
                        results_map[q_id].append(is_ambiguous)
                    else:
                        # Failed call
                        pass
                except Exception:
                    pass
                pbar.update(1)

    duration = time.time() - start_time
    print(f"\nCompleted in {duration:.2f} seconds (Avg {duration/total_tasks:.4f}s per call)")

    # Generate Summary
    print("\n" + "="*100)
    print(f"{'ID':<5} | {'Ambiguous %':<12} | {'Clear %':<10} | {'Total Valid':<12} | {'Query'}")
    print("="*100)
    
    summary_data = []

    for q_id, query in zip(ids, queries):
        results = results_map[q_id]
        total_valid = len(results)
        
        if total_valid == 0:
            print(f"{str(q_id):<5} | {'N/A':<12} | {'N/A':<10} | {0:<12} | {query[:60]}...")
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
            "total_runs": total_valid
        })

    # Save summary
    output_file = Path("stability_analysis_results.tsv")
    pd.DataFrame(summary_data).to_csv(output_file, sep="\t", index=False)
    print(f"\nDetailed summary saved to {output_file.absolute()}")

if __name__ == "__main__":
    main()
