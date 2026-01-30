#!/bin/bash
echo "Starting Parking LLM Service..."
uvicorn src.main:app --host 0.0.0.0 --port 7860
