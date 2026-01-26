#!/bin/bash
echo "Starting Parking LLM Service..."
uvicorn app:app --host 0.0.0.0 --port 8000
