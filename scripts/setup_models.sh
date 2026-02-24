#!/bin/bash
# ============================================================================
# Setup All Models for Universal Medical Ingestion Engine
# ============================================================================
# This script downloads and configures all required models:
# - MedGemma (LLM for text extraction)
# - VLM (Vision model for image extraction)
# - Embedding model (for vector store)
#
# Run this script after installing dependencies:
#   ./scripts/setup_models.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

echo "==================================================="
echo "Medical Ingestion Engine - Model Setup"
echo "==================================================="
echo ""

# Check if ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama is not running. Please start Ollama first:"
    echo "  ollama serve"
    exit 1
fi

echo "Ollama is running. Checking available models..."
echo ""

# ============================================================================
# 1. MedGemma (LLM for extraction)
# ============================================================================
echo "=== Step 1: MedGemma (LLM) ==="

# Check if medgemma-4b-local already exists
if ollama list | grep -q "medgemma-4b-local"; then
    echo "✓ MedGemma already installed (medgemma-4b-local)"
else
    echo "Setting up MedGemma from HuggingFace GGUF..."

    # Check if huggingface-cli is installed
    if ! command -v huggingface-cli &> /dev/null; then
        echo "Installing huggingface_hub..."
        pip install huggingface_hub
    fi

    mkdir -p "$MODELS_DIR"
    cd "$MODELS_DIR"

    # Download if not exists
    if [ ! -f "medgemma-4b-it-Q4_K_M.gguf" ]; then
        echo "Downloading MedGemma 4B GGUF (Q4_K_M - 2.49 GB)..."
        huggingface-cli download unsloth/medgemma-4b-it-GGUF \
            medgemma-4b-it-Q4_K_M.gguf \
            --local-dir . \
            --local-dir-use-symlinks False
    fi

    # Create Ollama model
    if [ -f "$MODELS_DIR/Modelfile.medgemma" ]; then
        echo "Creating Ollama model medgemma-4b-local..."
        ollama create medgemma-4b-local -f "$MODELS_DIR/Modelfile.medgemma"
        echo "✓ MedGemma installed successfully"
    else
        echo "ERROR: Modelfile.medgemma not found in $MODELS_DIR"
        exit 1
    fi
fi

echo ""

# ============================================================================
# 2. VLM (Vision Language Model)
# ============================================================================
echo "=== Step 2: VLM (Vision Model) ==="

# Moondream is smallest/fastest for CPU
VLM_MODEL="moondream"

if ollama list | grep -q "moondream"; then
    echo "✓ VLM already installed ($VLM_MODEL)"
else
    echo "Pulling VLM model ($VLM_MODEL)..."
    echo "This is a 1.8B parameter model, ideal for CPU inference."
    ollama pull $VLM_MODEL
    echo "✓ VLM installed successfully"
fi

echo ""

# ============================================================================
# 3. Embedding Model
# ============================================================================
echo "=== Step 3: Embedding Model ==="

# mxbai-embed-large is top performer on MTEB
EMBED_MODEL="mxbai-embed-large"

if ollama list | grep -q "mxbai-embed-large"; then
    echo "✓ Embedding model already installed ($EMBED_MODEL)"
else
    echo "Pulling embedding model ($EMBED_MODEL)..."
    echo "This is the top performer on MTEB benchmarks."
    ollama pull $EMBED_MODEL
    echo "✓ Embedding model installed successfully"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================
echo "==================================================="
echo "Setup Complete!"
echo "==================================================="
echo ""
echo "Installed models:"
ollama list | grep -E "medgemma|moondream|mxbai"
echo ""
echo "To use these models, ensure your .env file has:"
echo "  OLLAMA_MODEL=medgemma-4b-local"
echo "  VLM_MODEL=moondream"
echo "  EMBEDDING_MODEL=mxbai-embed-large"
echo "  EMBEDDING_DIM=1024"
echo ""
echo "Run a test with:"
echo "  python -c \"from src.medical_ingestion.core.config import get_config; print(get_config())\""
echo ""
