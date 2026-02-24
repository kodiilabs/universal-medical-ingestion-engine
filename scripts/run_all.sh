#!/bin/bash
# Run all services (Streamlit, FastAPI, React)
# Streamlit: port 8501
# FastAPI: port 8000
# React: port 3000

cd "$(dirname "$0")/.."

echo "Starting services..."

# Start Streamlit (existing UI)
echo "Starting Streamlit on port 8501..."
PYTHONPATH=src streamlit run ui/app.py --server.port 8501 &
STREAMLIT_PID=$!

# Start FastAPI backend
echo "Starting FastAPI on port 8000..."
PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Start React frontend
echo "Starting React frontend on port 3000..."
cd frontend && npm start &
REACT_PID=$!

echo ""
echo "Services started:"
echo "  - Streamlit UI: http://localhost:8501"
echo "  - FastAPI API:  http://localhost:8000"
echo "  - React UI:     http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "kill $STREAMLIT_PID $API_PID $REACT_PID 2>/dev/null; exit" INT TERM
wait
