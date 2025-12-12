#!/bin/bash
#
# Script to run evaluation on AmbigNQ and ClariQ datasets with few_shot and zero_shot strategies
#
# This script evaluates ambiguity classification performance on:
# - Datasets: ClariQ, AmbigNQ
# - Prompting methods: zero_shot, few_shot
#
# Usage: ./eval_datasets.sh

set -e  # Exit on error

# Configuration
BATCH_SIZE=32
MAX_WORKERS=8
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  Running Evaluation on AmbigNQ and ClariQ${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Batch Size: ${BATCH_SIZE}"
echo -e "  Max Workers: ${MAX_WORKERS}"
echo -e "  Results Directory: ${RESULTS_DIR}"
echo -e "  Timestamp: ${TIMESTAMP}"
echo ""

# Create results directories
mkdir -p "$RESULTS_DIR/clariq"
mkdir -p "$RESULTS_DIR/ambignq"

# Define datasets and prompting methods
DATASETS=("clariq" "ambignq")
STRATEGIES=("zero_shot" "few_shot")

# Track success/failure
TOTAL_RUNS=$((${#DATASETS[@]} * ${#STRATEGIES[@]}))
COMPLETED_RUNS=0
FAILED_RUNS=0

echo -e "${BLUE}Total evaluations to run: ${TOTAL_RUNS}${NC}"
echo ""

# Main evaluation loop
for dataset in "${DATASETS[@]}"; do
    echo -e "${YELLOW}======================================================================${NC}"
    echo -e "${YELLOW}  Dataset: ${dataset^^}${NC}"
    echo -e "${YELLOW}======================================================================${NC}"
    echo ""
    
    for strategy in "${STRATEGIES[@]}"; do
        RUN_NUM=$((COMPLETED_RUNS + FAILED_RUNS + 1))
        echo -e "${GREEN}[${RUN_NUM}/${TOTAL_RUNS}] Running evaluation: ${dataset} - ${strategy}${NC}"
        echo -e "  Strategy: ${strategy}"
        echo -e "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"
        
        # Set output paths
        OUTPUT_FILE="${RESULTS_DIR}/${dataset}/results_${dataset}_${strategy}_${TIMESTAMP}.tsv"
        
        # Run evaluation
        if python "$SCRIPT_DIR/evaluate_ambiguity_classification.py" \
            --dataset "$dataset" \
            --strategy "$strategy" \
            --batch-size "$BATCH_SIZE" \
            --max-workers "$MAX_WORKERS" \
            --output "$OUTPUT_FILE"; then
            
            COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
            echo -e "${GREEN}  ✓ Success!${NC}"
            echo -e "  Results saved to: ${OUTPUT_FILE}"
        else
            FAILED_RUNS=$((FAILED_RUNS + 1))
            echo -e "${RED}  ✗ Failed!${NC}"
        fi
        
        echo -e "  Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
    done
    
    echo ""
done

# Print summary
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  Evaluation Summary${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}Completed: ${COMPLETED_RUNS}/${TOTAL_RUNS}${NC}"
if [ $FAILED_RUNS -gt 0 ]; then
    echo -e "${RED}Failed: ${FAILED_RUNS}/${TOTAL_RUNS}${NC}"
fi
echo ""
echo -e "${GREEN}Results saved to: ${RESULTS_DIR}${NC}"
echo -e "  - ClariQ results: ${RESULTS_DIR}/clariq/"
echo -e "  - AmbigNQ results: ${RESULTS_DIR}/ambignq/"
echo ""

# Create summary file listing all results
SUMMARY_FILE="${RESULTS_DIR}/evaluation_summary_${TIMESTAMP}.txt"
cat > "$SUMMARY_FILE" << EOF
Ambiguity Classification Evaluation Summary
Generated: $(date '+%Y-%m-%d %H:%M:%S')

Configuration:
- Batch Size: ${BATCH_SIZE}
- Max Workers: ${MAX_WORKERS}
- Timestamp: ${TIMESTAMP}

Results:
- Total Runs: ${TOTAL_RUNS}
- Completed: ${COMPLETED_RUNS}
- Failed: ${FAILED_RUNS}

Output Files:
EOF

# List all generated files
for dataset in "${DATASETS[@]}"; do
    echo "" >> "$SUMMARY_FILE"
    echo "${dataset^^}:" >> "$SUMMARY_FILE"
    for strategy in "${STRATEGIES[@]}"; do
        result_file="${RESULTS_DIR}/${dataset}/results_${dataset}_${strategy}_${TIMESTAMP}.tsv"
        metrics_file="${RESULTS_DIR}/${dataset}/results_${dataset}_${strategy}_${TIMESTAMP}_metrics.json"
        if [ -f "$result_file" ]; then
            echo "  ✓ ${strategy}: ${result_file}" >> "$SUMMARY_FILE"
            if [ -f "$metrics_file" ]; then
                echo "    Metrics: ${metrics_file}" >> "$SUMMARY_FILE"
            fi
        else
            echo "  ✗ ${strategy}: Not found" >> "$SUMMARY_FILE"
        fi
    done
done

echo -e "${BLUE}Summary saved to: ${SUMMARY_FILE}${NC}"
echo ""

if [ $FAILED_RUNS -eq 0 ]; then
    echo -e "${GREEN}✓ All evaluations completed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Some evaluations failed. Check the logs above for details.${NC}"
    exit 1
fi