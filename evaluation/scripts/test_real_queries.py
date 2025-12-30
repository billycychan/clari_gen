
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
from core.clari_gen.clients import SmallModelClient
from core.clari_gen.prompts import BinaryDetectionPrompt
from core.clari_gen.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

def main():
    # Load queries
    input_file = Path("real-queries.tsv")
    print(f"Loading queries from {input_file}...")
    
    # Check if file exists
    if not input_file.exists():
        print(f"Error: {input_file} does not exist.")
        return

    # Read the file
    # Assuming the file format is: ID <tab> Query (no header)
    try:
        df = pd.read_csv(input_file, sep="\t", header=None, names=["id", "query"])
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    queries = df["query"].tolist()
    ids = df["id"].tolist()
    
    print(f"Found {len(queries)} queries.")

    # Initialize client
    print("Initializing model client...")
    try:
        client = SmallModelClient()
        if not client.test_connection():
             print("Error: Could not connect to model server.")
             print("Please ensure the model server is running (cd server && ./serve_models.sh)")
             return
    except Exception as e:
        print(f"Error initializing client: {e}")
        return

    results = []
    
    print("\nStarting ambiguity detection...\n")
    # Print table header
    print(f"{'ID':<5} | {'Status':<10} | {'Query'}")
    print("-" * 100)

    for i, (q_id, query) in enumerate(zip(ids, queries)):
        try:
            # Create messages with few-shot strategy
            messages = BinaryDetectionPrompt.create_messages(query, strategy="zero_shot")
            
            # Call model
            response = client.detect_binary_ambiguity(
                messages,
                response_format=BinaryDetectionPrompt.get_response_schema(),
            )
            
            # Parse response
            data = BinaryDetectionPrompt.parse_response(response)
            is_ambiguous = data["is_ambiguous"]
            status = "AMBIGUOUS" if is_ambiguous else "CLEAR"
            
            # Store result
            results.append({
                "id": q_id,
                "query": query,
                "is_ambiguous": is_ambiguous,
                "status": status
            })
            
            # Print row
            # Truncate query for display
            display_query = (query[:75] + '...') if len(query) > 75 else query
            print(f"{str(q_id):<5} | {status:<10} | {display_query}")
            
        except Exception as e:
            print(f"{str(q_id):<5} | {'ERROR':<10} | {query[:60]}... ({str(e)})")
            results.append({
                "id": q_id,
                "query": query,
                "is_ambiguous": None,
                "status": "ERROR",
                "error": str(e)
            })

    # Save results to TSV
    output_file = Path("real_queries_results.tsv")
    pd.DataFrame(results).to_csv(output_file, sep="\t", index=False)
    print(f"\nResults saved to {output_file.absolute()}")

if __name__ == "__main__":
    main()
