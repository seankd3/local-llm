#!/bin/bash
# Startup script for local-llm

cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Run the application
python -m src.main "$@"
