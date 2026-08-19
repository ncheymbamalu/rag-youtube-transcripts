#!/bin/bash
# Set the project's home directory and log file
BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$(dirname "${BIN_DIR}")" && pwd)"
LOGS_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOGS_DIR}/cron_$(date +'%Y_%m_%d_%H_%M_%S').log"

# Define the absolute path to the .env file, which contains the `GIT` and `MAKE` variables
ENV_FILE="${PROJECT_DIR}/.env"

# Check if the .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

# Load and export all variables from the .env file
set -a
source "$ENV_FILE"
set +a

# Set the paths for the Python executable, DVC executable, and ETL pipeline script
PYTHON="${PROJECT_DIR}/.venv/bin/python"
DVC="${PROJECT_DIR}/.venv/bin/dvc"
SCRIPT="${PROJECT_DIR}/src/rag_youtube_transcripts/pipelines/etl.py"

# Change directories to the project's home directory and ensure the logs directory exists
cd "$PROJECT_DIR" || exit 1
mkdir -p "$LOGS_DIR"

# Route everything below this line to the log file
# NOTE: on an interactive run stdout is a terminal, so the output is mirrored there as well;
# under cron it is not, so the log is the only record
if [ -t 1 ]; then
    exec > >(tee -a "$LOG_FILE") 2>&1
else
    exec >> "$LOG_FILE" 2>&1
fi

# Prune cron logs older than a week
find "${LOGS_DIR}" -name 'cron_*.log' -type f -mtime +6 -delete

# Pull the current artifacts
# NOTE: if the artifacts can't be retrieved, then the pipeline will not be executed
if ! $DVC pull; then
    echo "dvc pull failed - aborting before the pipeline runs."
    exit 1
fi

# Execute the pipeline
# NOTE: the artifacts are only committed if the pipeline was successful and all three of
# `embeddings.parquet`, `bm25.parquet`, and `transcripts.parquet` were updated
if $PYTHON "$SCRIPT"; then

    # Check for changes in the artifacts directory
    $DVC status --quiet || export CHANGES="./artifacts/ has been modified."

    # If there were changes, commit and push them to DVC and Git/GitHub
    printenv CHANGES && \
    $DVC add ./artifacts && \
    $GIT add artifacts.dvc && \
    $GIT commit -m "Executing the ETL pipeline and updating ./artifacts.dvc" && \
    $DVC push && \
    $GIT push

else
    echo "ETL failed (exit $?) - artifacts left uncommitted."
fi

# Clean up regardless of the pipeline's outcome
$MAKE clean

# Delete the CHANGES environment variable
unset CHANGES
