#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists (uncomment and modify if you have one)
# source /path/to/your/venv/bin/activate

# Set up logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ingestion_$(date +\%Y\%m\%d_\%H\%M\%S).log"

# Run the ingestion script and log output
echo "Starting data ingestion at $(date)" >> "$LOG_FILE"
python3 data_ingestion_service.py >> "$LOG_FILE" 2>&1
echo "Finished data ingestion at $(date)" >> "$LOG_FILE"

# Optional: Clean up old log files (keep last 7 days)
find "$LOG_DIR" -name "ingestion_*.log" -mtime +7 -delete 