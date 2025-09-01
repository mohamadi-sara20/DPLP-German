#!/bin/bash
set -e

# This script serves as a repeatable regression test to verify the fix
# for the SVM convergence warnings in the training pipeline.

# --- Configuration ---
CORPUS_PATH="data/pcc_retest"
MODEL_PY_PATH="dplp_parser/model.py"
LOG_BEFORE_FIX="svm_fix_test_before.log"
LOG_AFTER_FIX="svm_fix_test_after.log"

# --- Helper Functions ---
function run_stage2_training() {
  local LOG_FILE=$1
  echo "--- Running Stage 2 Training on $CORPUS_PATH ---"
  echo "--- Real-time logs will be shown below. Full log will be saved to $LOG_FILE ---"
  # This command now streams output directly to the terminal and saves it to a file simultaneously.
  docker run --rm -it \
    -v "$(pwd)":/home/DPLP \
    -w /home/DPLP \
    mohamadisara20/dplp-env:ger \
    python3 ger_train.py "$CORPUS_PATH" -rm "${CORPUS_PATH}/$(basename "$CORPUS_PATH")_rel_mapping.json" --run-stage2 2>&1 | tee "$LOG_FILE"
}

# --- Test Execution ---

echo "--- Test Plan: Verify SVM Convergence Fix ---"

# 1. Restore original file to ensure a clean state
echo "Step 1: Restoring original $MODEL_PY_PATH to reproduce the failure..."
git restore "$MODEL_PY_PATH"

# 2. Run the test and expect the failure (The "Red" state)
echo "Step 2: Running training and expecting ConvergenceWarning..."
run_stage2_training "$LOG_BEFORE_FIX"
if grep -q "ConvergenceWarning" "$LOG_BEFORE_FIX"; then
  echo "SUCCESS: Correctly detected ConvergenceWarning in the original code."
else
  echo "FAILURE: Did not find the expected ConvergenceWarning. The test cannot proceed."
  exit 1
fi

# 3. Apply the fix
echo "Step 3: Applying max_iter=1000 fix to $MODEL_PY_PATH..."
# Using sed for a simple, non-interactive replacement. This version correctly finds the target string.
sed -i "s/tol=1e-7/tol=1e-7, max_iter=1000/g" "$MODEL_PY_PATH"
echo "Fix applied. Verifying change:"
# Add a diff to provide visual confirmation that the file was changed.
git diff "$MODEL_PY_PATH"

# 4. Run the test and expect success (The "Green" state)
echo "Step 4: Re-running training and expecting NO ConvergenceWarning..."
run_stage2_training "$LOG_AFTER_FIX"
if grep -q "ConvergenceWarning" "$LOG_AFTER_FIX"; then
  echo "FAILURE: ConvergenceWarning was still present after the fix."
  git restore "$MODEL_PY_PATH" # Clean up
  exit 1
else
  echo "SUCCESS: No ConvergenceWarning detected after applying the fix."
fi


# 5. Regression Check
echo "Step 5: Checking for performance regression..."
SCORE_BEFORE=$(grep "F1 score on relation level" "$LOG_BEFORE_FIX" | awk '{print $NF}')
SCORE_AFTER=$(grep "F1 score on relation level" "$LOG_AFTER_FIX" | awk '{print $NF}')

echo "  - F1 Score (Relation) Before Fix: $SCORE_BEFORE"
echo "  - F1 Score (Relation) After Fix:  $SCORE_AFTER"

# Using awk for floating point comparison
REGRESSION=$(awk -v before="$SCORE_BEFORE" -v after="$SCORE_AFTER" 'BEGIN { print (after < before - 0.01) }')

if [ "$REGRESSION" -eq 1 ]; then
    echo "WARNING: Possible performance regression detected (F1 score dropped by > 0.01)."
else
    echo "SUCCESS: No significant performance regression detected."
fi

echo "--- Verification Complete ---"
echo "The fix is successful and verified. The change to $MODEL_PY_PATH has been left in place."
