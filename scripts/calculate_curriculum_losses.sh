#!/bin/bash

# Default values
MODEL_PATH=""
DATA_CFG_PATH=""
DATASET_SPLIT=""
OUTPUT_PATH=""
BATCH_SIZE=4
NUM_WORKERS=0

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --data_cfg_path) DATA_CFG_PATH="$2"; shift ;;
        --dataset_split) DATASET_SPLIT="$2"; shift ;;
        --output_path) OUTPUT_PATH="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --num_workers) NUM_WORKERS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check required arguments
if [[ -z "$MODEL_PATH" ]] || [[ -z "$DATA_CFG_PATH" ]] || [[ -z "$DATASET_SPLIT" ]] || [[ -z "$OUTPUT_PATH" ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: $0 --model_path <path> --data_cfg_path <path> --dataset_split <split> --output_path <path> [--batch_size <int>] [--num_workers <int>]"
    exit 1
fi

echo "Running curriculum loss calculation..."
echo "  Model Path: $MODEL_PATH"
echo "  Data Config: $DATA_CFG_PATH"
echo "  Dataset Split: $DATASET_SPLIT"
echo "  Output Path: $OUTPUT_PATH"
echo "  Batch Size: $BATCH_SIZE"
echo "  Num Workers: $NUM_WORKERS"

# Determine the directory of the current script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_SCRIPT_PATH="$SCRIPT_DIR/calculate_curriculum_losses_py.py"
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/..") # Assumes script is in project_root/scripts

# Ensure Python script exists
if [[ ! -f "$PYTHON_SCRIPT_PATH" ]]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT_PATH"
    exit 1
fi

# Set PYTHONPATH to include the project root src directory
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"


# Execute the Python script
# Ensure you have activated the correct conda/virtual environment before running this bash script
python "$PYTHON_SCRIPT_PATH" \
    --model_path "$MODEL_PATH" \
    --data_cfg_path "$DATA_CFG_PATH" \
    --dataset_split "$DATASET_SPLIT" \
    --output_path "$OUTPUT_PATH" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS"

# Check the exit status of the Python script
if [[ $? -ne 0 ]]; then
    echo "Error: Python script failed to execute successfully."
    exit 1
fi

echo "Python script finished."
