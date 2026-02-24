#!/bin/bash
# Setup MedGemma 4B from HuggingFace for local CPU inference via Ollama
#
# This script downloads the quantized MedGemma model from HuggingFace
# and creates an Ollama model for local inference without GPU.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

echo "==================================================="
echo "MedGemma 4B Local Setup (HuggingFace -> Ollama)"
echo "==================================================="
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Check if ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama is not running. Please start Ollama first:"
    echo "  ollama serve"
    exit 1
fi

echo "Step 1: Creating models directory..."
mkdir -p "$MODELS_DIR"

echo ""
echo "Step 2: Downloading MedGemma 4B GGUF (Q4_K_M - 2.49 GB)..."
echo "This may take a few minutes depending on your connection..."
echo ""

cd "$MODELS_DIR"

# Download the Q4_K_M quantization (best for CPU without GPU)
huggingface-cli download unsloth/medgemma-4b-it-GGUF \
    medgemma-4b-it-Q4_K_M.gguf \
    --local-dir . \
    --local-dir-use-symlinks False

echo ""
echo "Step 3: Creating Ollama model..."
echo ""

# Create the Ollama model from the Modelfile
ollama create medgemma-4b-local -f "$MODELS_DIR/Modelfile.medgemma"

echo ""
echo "==================================================="
echo "Setup Complete!"
echo "==================================================="
echo ""
echo "The model is now available as: medgemma-4b-local"
echo ""
echo "Test it with:"
echo "  ollama run medgemma-4b-local \"What is the normal fasting glucose range?\""
echo ""
echo "To use in your .env file, set:"
echo "  OLLAMA_MODEL=medgemma-4b-local"
echo ""
