#!/bin/bash
# ============================================================================
# Download OCR models for Medical Ingestion Engine
# Primary: TrOCR (robust on degraded/faxed documents)
# Optional: PaddleOCR (fast fallback for clean digital PDFs)
# ============================================================================

set -e

echo "=== Medical Ingestion Engine - Model Setup ==="
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Not in a virtual environment. Activate your venv first."
    echo "  source .venv/bin/activate"
    echo ""
fi

# Fix SSL certificates on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "=== Fixing SSL certificates for macOS ==="

    # Install certifi
    pip install --quiet certifi

    # Get the certificate path
    CERT_PATH=$(python -c "import certifi; print(certifi.where())")

    echo "Setting SSL_CERT_FILE=$CERT_PATH"
    export SSL_CERT_FILE="$CERT_PATH"
    export REQUESTS_CA_BUNDLE="$CERT_PATH"

    # Add to shell profile if not already there
    SHELL_PROFILE=""
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_PROFILE="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_PROFILE="$HOME/.bashrc"
    fi

    if [ -n "$SHELL_PROFILE" ]; then
        if ! grep -q "SSL_CERT_FILE" "$SHELL_PROFILE"; then
            echo ""
            echo "Adding SSL environment variables to $SHELL_PROFILE"
            echo "" >> "$SHELL_PROFILE"
            echo "# SSL certificates for Python" >> "$SHELL_PROFILE"
            echo "export SSL_CERT_FILE=\"\$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || echo '')\"" >> "$SHELL_PROFILE"
            echo "export REQUESTS_CA_BUNDLE=\"\$SSL_CERT_FILE\"" >> "$SHELL_PROFILE"
            echo "Added! Run: source $SHELL_PROFILE"
        fi
    fi

    echo ""
fi

# =============================================================================
# REQUIRED: TrOCR (Primary OCR for all text)
# =============================================================================
echo "=== Downloading TrOCR models (PRIMARY OCR - REQUIRED) ==="
echo "TrOCR is used for ALL text - robust on degraded/faxed documents"
echo ""

python << 'EOF'
import os
import sys

try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
except ImportError:
    pass

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    print("Downloading TrOCR model (microsoft/trocr-base-handwritten)...")
    print("This is the PRIMARY OCR engine for all text regions.")
    model_name = "microsoft/trocr-base-handwritten"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    print(f"\n✓ TrOCR model downloaded successfully!")
    print("  Model: microsoft/trocr-base-handwritten")
    print("  Usage: All text (printed, handwriting, tables)")
except ImportError:
    print("✗ Transformers not installed!")
    print("  Install with: pip install transformers torch")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ TrOCR download failed: {e}")
    sys.exit(1)
EOF

# =============================================================================
# OPTIONAL: PaddleOCR (Fast fallback for clean digital PDFs)
# =============================================================================
echo ""
echo "=== PaddleOCR (OPTIONAL - fast OCR for clean digital PDFs) ==="
echo "PaddleOCR is only used when skip_preprocessing=True (clean PDFs)"
read -p "Download PaddleOCR/EasyOCR? (y/N): " download_paddle
if [[ "$download_paddle" =~ ^[Yy]$ ]]; then
    python << 'EOF'
import os
import sys

try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
except ImportError:
    pass

try:
    import easyocr
    print("Initializing EasyOCR and downloading models...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=True)
    print("\n✓ EasyOCR models downloaded successfully!")
except Exception as e:
    print(f"\n⚠ EasyOCR download failed: {e}")
    print("  This is optional - TrOCR will handle all OCR tasks.")
EOF
else
    echo "Skipping PaddleOCR/EasyOCR (TrOCR will handle all OCR)"
fi

echo ""
echo "=== Downloading PaliGemma (optional - VLM for region classification) ==="
echo "Note: PaliGemma requires ~3GB and HuggingFace authentication"
read -p "Download PaliGemma? (y/N): " download_paligemma
if [[ "$download_paligemma" =~ ^[Yy]$ ]]; then
    python << 'EOF'
import os
import sys

try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
except ImportError:
    pass

try:
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    print("Downloading PaliGemma model (this may take a while)...")
    model_name = "google/paligemma-3b-pt-224"
    processor = AutoProcessor.from_pretrained(model_name)
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_name)
    print(f"\n✓ PaliGemma model downloaded: {model_name}")
except ImportError:
    print("✗ Transformers not installed. Install with: pip install transformers")
except Exception as e:
    print(f"\n✗ PaliGemma download failed: {e}")
    print("  You may need to: huggingface-cli login")
EOF
else
    echo "Skipping PaliGemma download"
fi

echo ""
echo "=== Checking Tesseract (fallback OCR) ==="
if command -v tesseract &> /dev/null; then
    TESS_VERSION=$(tesseract --version 2>&1 | head -n1)
    echo "✓ Tesseract installed: $TESS_VERSION"
else
    echo "✗ Tesseract not installed (optional but recommended)"
    echo "  Install with: brew install tesseract"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Models installed:"
echo ""
echo "  REQUIRED (Core OCR):"
echo "  ─────────────────────────────────────────────────────────"
echo "  • TrOCR         : PRIMARY OCR for all text"
echo "                    Robust on degraded/faxed documents"
echo "                    Model: microsoft/trocr-base-handwritten"
echo ""
echo "  OPTIONAL:"
echo "  ─────────────────────────────────────────────────────────"
echo "  • PaddleOCR     : Fast OCR for clean digital PDFs"
echo "                    Only used with skip_preprocessing=True"
echo "  • PaliGemma     : VLM for region classification"
echo "  • MedGemma      : Medical understanding (via Ollama)"
echo ""
echo "To use MedGemma, ensure Ollama is running:"
echo "  ollama serve"
echo "  ollama pull medgemma"
echo ""
echo "OCR Routing:"
echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │ Region Type    │ OCR Engine                        │"
echo "  ├─────────────────────────────────────────────────────┤"
echo "  │ printed_text   │ TrOCR (robust to noise/skew)      │"
echo "  │ handwriting    │ TrOCR                             │"
echo "  │ table          │ TrOCR + cell detection            │"
echo "  │ stamp/signature│ SKIP (non-text)                   │"
echo "  └─────────────────────────────────────────────────────┘"
echo ""
