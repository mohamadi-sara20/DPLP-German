#!/bin/bash

# This script trains a new DPLP model for German.

# Prerequisites:
# 1. Docker is installed and running.
# 2. The DPLP-German docker image is available (mohamadisara20/dplp-env:ger).
# 3. Your training data is prepared in the required format.

# Instructions:
# 1. Set the BASE_DIR variable below to the absolute path of your corpus directory.
#    This directory must contain 'training', 'dev', and 'test' subdirectories
#    filled with your .rs3 files.
# 2. Make sure your parsing_eval_metrics/rel_mapping.json file is up-to-date
#    with all the relations in your corpus.
# 3. Run this script from the root of the DPLP-German project directory:
#    bash train_custom_model.sh

# --- Argument Validation ---
if [ -z "$1" ]; then
  echo "Usage: $0 <corpus_base_path>"
  echo "Example: $0 data/pcc"
  exit 1
fi

# --- Configuration ---
# The main directory for the corpus, passed as a command-line argument.
BASE_DIR="$1"
# Derive the corpus name and relation map file from the base directory path.
CORPUS_NAME=$(basename "$BASE_DIR")
REL_MAP_FILE="${BASE_DIR}/${CORPUS_NAME}_rel_mapping.json"

# --- Script ---

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
  echo "Error: Base directory '$BASE_DIR' not found."
  echo "Please ensure it contains 'training', 'dev', and 'test' subdirectories."
  exit 1
fi

echo "--- Step 1: Generating custom relation map for corpus: $CORPUS_NAME ---"
python3 scripts/generate_relation_map.py "${BASE_DIR}/training" "${REL_MAP_FILE}"
if [ $? -ne 0 ]; then
    echo "Error: Failed to generate relation map. Aborting."
    exit 1
fi

echo "--- Step 2: Running Stage 1 Preprocessing (in Docker) ---"
# This stage generates the .txt files from the source .rs3 files.
docker run --rm -it \
  -v "$(pwd)":/home/DPLP \
  -w /home/DPLP \
  mohamadisara20/dplp-env:ger \
  python3 ger_train.py "$BASE_DIR" --run-stage1

echo "--- Step 3: Running Host-side GPU Preprocessing ---"
# This stage uses the host's GPU for Stanza and also runs the BerkeleyParser.
for CORPUS_PART in training dev test;
do
    echo "--- Processing ./${BASE_DIR}/${CORPUS_PART} on host ---"
    python3 run_stanza_preprocessing.py "${BASE_DIR}/${CORPUS_PART}"
    python3 ger_4_txt2parse.py "${BASE_DIR}/${CORPUS_PART}"
done

echo "--- Step 4: Running Stage 2 Final Training (in Docker) ---"
# This stage takes the preprocessed files and runs the final training steps.
docker run --rm -it \
  -v "$(pwd)":/home/DPLP \
  -w /home/DPLP \
  mohamadisara20/dplp-env:ger \
  python3 ger_train.py "$BASE_DIR" -rm "${REL_MAP_FILE}" --run-stage2


echo "Training finished."
echo "Your new model is located in: $BASE_DIR/model/"
echo "Training results are in: $BASE_DIR/result.txt"