#!/bin/bash
# 1. Set the project's home directory and log file
BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$(dirname "${BIN_DIR}")" && pwd)"
LOGS_DIR=${PROJECT_DIR}/logs
LOG_FILE="${LOGS_DIR}/cron_$(date +'%Y_%m_%d_%H_%M_%S').log"

# 2. Define the absolute path to the .env file, which contains the `GIT` and `MAKE` variables
ENV_FILE=${PROJECT_DIR}/.env

# 3. Check if the .env file exists
if [ ! -f $ENV_FILE ]; then
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

# 4. Load and export all variables from the .env file
set -a
source $ENV_FILE
set +a

# 5. Set the paths for the Python executable, DVC executable, and ETL pipeline script
PYTHON=${PROJECT_DIR}/.venv/bin/python
DVC=${PROJECT_DIR}/.venv/bin/dvc
SCRIPT=${PROJECT_DIR}/src/rag_youtube_transcripts/pipelines/etl.py

# 6. Change directories to the project's home directory and ensure the logs directory exists
cd $PROJECT_DIR || exit 1
mkdir -p $LOGS_DIR

# 7. Execute the pipeline
$DVC pull
$PYTHON $SCRIPT >> $LOG_FILE 2>&1 ; $MAKE clean

# 8. Check for changes in the artifacts directory
$DVC status --quiet || export CHANGES="./artifacts/ has been modified."

# 9. If there were changes, commit and push them to DVC and Git/GitHub
printenv CHANGES && \
$DVC add ./artifacts && \
$GIT add artifacts.dvc && \
$GIT commit -m "Executing the ETL pipeline and updating ./artifacts.dvc" && \
$DVC push && \
$GIT push

# 10. Delete the CHANGES environment variable
unset CHANGES
