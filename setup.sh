#!/usr/bin/env bash
set -euo pipefail

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
NC="\033[0m"

echo -e "${BOLD}${CYAN}=== Gemma-Talk: Local Hands-Free Voice Assistant Bootstrap ===${NC}\n"

# 1. Environment & Config Loading
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
    echo -e "${GREEN}✓ Loading .env configuration${NC}"
    # Export non-comment lines
    set -a
    source .env
    set +a
else
    echo -e "${YELLOW}! .env not found, copying from .env.example${NC}"
    cp .env.example .env
    set -a
    source .env
    set +a
fi

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4}"

# 2. Check Prerequisite Tools
echo -e "\n${BOLD}[1/5] Checking Tooling Prerequisites...${NC}"

if ! command -v node &> /dev/null; then
    echo -e "${RED}✗ Node.js is required but not found in PATH.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js detected: $(node -v)${NC}"

if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}! uv package manager not found. Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo -e "${GREEN}✓ uv detected: $(uv --version)${NC}"

# 3. Verify Ollama & Pull LLM Model
echo -e "\n${BOLD}[2/5] Verifying Ollama Runtime & Model...${NC}"
echo -e "Probing Ollama endpoint at ${CYAN}${OLLAMA_BASE_URL}${NC}..."

if ! curl -sf "${OLLAMA_BASE_URL}/api/tags" &> /dev/null; then
    echo -e "${RED}✗ Cannot reach Ollama at ${OLLAMA_BASE_URL}.${NC}"
    echo -e "${YELLOW}Please ensure Ollama is running (e.g., 'ollama serve' in another terminal) and re-run setup.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Ollama server is running and accessible.${NC}"

echo -e "Checking if model '${OLLAMA_MODEL}' is pulled..."
TAGS_JSON=$(curl -s "${OLLAMA_BASE_URL}/api/tags")
if echo "$TAGS_JSON" | grep -q "\"${OLLAMA_MODEL}\""; then
    echo -e "${GREEN}✓ Model '${OLLAMA_MODEL}' is already available in Ollama.${NC}"
else
    echo -e "${YELLOW}Pulling Ollama model '${OLLAMA_MODEL}' (this may take a couple minutes)...${NC}"
    ollama pull "${OLLAMA_MODEL}"
    echo -e "${GREEN}✓ Successfully pulled '${OLLAMA_MODEL}'.${NC}"
fi

# 4. Download Voice and VAD ONNX Weights
echo -e "\n${BOLD}[3/5] Setting up Local Audio Model Weights...${NC}"
MODELS_DIR="${SCRIPT_DIR}/server/models"
mkdir -p "${MODELS_DIR}"

# Silero VAD ONNX (~2.2MB)
SILERO_FILE="${MODELS_DIR}/silero_vad.onnx"
if [ ! -f "$SILERO_FILE" ]; then
    echo -e "Downloading Silero VAD v5 ONNX model..."
    curl -L --retry 3 --progress-bar \
        "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx" \
        -o "${SILERO_FILE}"
    echo -e "${GREEN}✓ Saved silero_vad.onnx (${SILERO_FILE})${NC}"
else
    echo -e "${GREEN}✓ silero_vad.onnx exists.${NC}"
fi

# Kokoro-82M ONNX model (~310MB)
KOKORO_FILE="${MODELS_DIR}/kokoro-v1.0.onnx"
if [ ! -f "$KOKORO_FILE" ]; then
    echo -e "Downloading Kokoro-82M v1.0 ONNX weights (~310MB)..."
    curl -L --retry 3 --progress-bar \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" \
        -o "${KOKORO_FILE}"
    echo -e "${GREEN}✓ Saved kokoro-v1.0.onnx${NC}"
else
    echo -e "${GREEN}✓ kokoro-v1.0.onnx exists.${NC}"
fi

# Kokoro Voices Binary (~27MB)
VOICES_FILE="${MODELS_DIR}/voices-v1.0.bin"
if [ ! -f "$VOICES_FILE" ]; then
    echo -e "Downloading Kokoro voice embeddings (voices-v1.0.bin ~27MB)..."
    curl -L --retry 3 --progress-bar \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" \
        -o "${VOICES_FILE}"
    echo -e "${GREEN}✓ Saved voices-v1.0.bin${NC}"
else
    echo -e "${GREEN}✓ voices-v1.0.bin exists.${NC}"
fi

# 5. Install Server Python Virtual Environment
echo -e "\n${BOLD}[4/5] Installing Backend Python Environment...${NC}"
cd "${SCRIPT_DIR}/server"
if [ ! -d ".venv" ]; then
    echo -e "Creating virtual environment using uv (Python 3.11)..."
    uv venv --python 3.11 .venv
fi
echo -e "Installing server dependencies..."
VIRTUAL_ENV="${SCRIPT_DIR}/server/.venv" uv pip install -e .
echo -e "${GREEN}✓ Backend dependencies installed successfully.${NC}"

# 6. Install Client Dependencies
echo -e "\n${BOLD}[5/5] Installing Frontend Web Client Dependencies...${NC}"
cd "${SCRIPT_DIR}/client"
if [ ! -d "node_modules" ]; then
    echo -e "Running npm install..."
    npm install
else
    echo -e "${GREEN}✓ node_modules already installed.${NC}"
fi

echo -e "\n${BOLD}${GREEN}===================================================================${NC}"
echo -e "${BOLD}${GREEN}✓ Phase 2 Setup Completed Successfully!${NC}"
echo -e "All models, voice files, Ollama configuration, and dependencies are ready."
echo -e "${BOLD}${GREEN}===================================================================${NC}\n"
