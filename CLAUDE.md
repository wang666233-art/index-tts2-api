# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IndexTTS-vLLM is an optimized implementation of the Index-TTS text-to-speech system that uses vLLM to accelerate GPT model inference. The project supports three versions of IndexTTS:
- Index-TTS v1.0
- IndexTTS v1.5
- IndexTTS v2.0

The system achieves significant performance improvements on RTX 4090:
- RTF improvement: ~0.3 -> ~0.1
- GPT decode speed: ~90 tokens/s -> ~280 tokens/s
- Concurrency: ~16 concurrent requests with 25% GPU memory utilization

## Core Architecture

### Main Components
- **`indextts/infer_vllm.py`** - Core inference engine using vLLM for v1/v1.5
- **`indextts/infer_vllm_v2.py`** - Inference engine for v2.0
- **`indextts/gpt/`** - GPT model implementations with vLLM integration
- **`indextts/s2mel/`** - Speech-to-mel spectrogram conversion modules
- **`indextts/BigVGAN/`** - Neural vocoder for audio generation
- **`webui.py`** / **`webui_v2.py`** - Gradio web interfaces
- **`api_server.py`** / **`api_server_v2.py`** - FastAPI REST API servers

### Model Architecture
The system follows a three-stage pipeline:
1. **Text Processing** - Text normalization and tokenization
2. **GPT-based TTS** - Uses vLLM-accelerated GPT models for sequence generation
3. **Audio Synthesis** - Converts mel spectrograms to waveform using BigVGAN

### Key Features
- **Multi-reference Audio Mixing** - Support for multiple reference audio files for voice styling
- **Speaker Registry** - Pre-defined speaker configurations in `assets/speaker.json`
- **OpenAI API Compatibility** - Compatible with OpenAI's `/audio/speech` endpoint
- **Concurrent Processing** - vLLM enables efficient batch processing

## Common Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n index-tts-vllm python=3.12
conda activate index-tts-vllm

# Install PyTorch 2.8.0 (required for vllm 0.10.2)
# Visit https://pytorch.org/get-started/locally/ for specific commands

# Install dependencies
pip install -r requirements.txt
```

### Model Management
```bash
# Download models using ModelScope
modelscope download --model kusuriuri/Index-TTS-vLLM --local_dir ./checkpoints/Index-TTS-vLLM
modelscope download --model kusuriuri/Index-TTS-1.5-vLLM --local_dir ./checkpoints/Index-TTS-1.5-vLLM
modelscope download --model kusuriuri/IndexTTS-2-vLLM --local_dir ./checkpoints/IndexTTS-2-vLLM

# Convert custom model weights (optional)
bash convert_hf_format.sh /path/to/your/model_dir
```

### Running Applications

#### Web Interface
```bash
# Index-TTS v1.0
python webui.py

# Index-TTS v1.5
python webui.py --version 1.5

# Index-TTS v2.0
python webui_v2.py
```

#### API Server
```bash
# Index-TTS v1.0/1.5
python api_server.py --model_dir ./checkpoints/Index-TTS-vLLM

# Index-TTS v2.0
python api_server_v2.py --model_dir ./checkpoints/IndexTTS-2-vLLM
```

#### API Server Parameters
- `--model_dir`: Path to model weights (required)
- `--host`: Server IP (default: 0.0.0.0)
- `--port`: Server port (default: 6006)
- `--gpu_memory_utilization`: vLLM GPU memory utilization (default: 0.25)

### Testing and Development
```bash
# Test API with provided examples
python api_example.py  # For v1/v1.5
python api_example_v2.py  # For v2.0

# Performance testing (requires API server to be running)
python simple_test.py  # Concurrency testing script
```

## Model Version Differences

### v1.0 vs v1.5
- Different model architectures and training data
- Both use the same inference pipeline (`infer_vllm.py`)
- Support multi-reference audio mixing

### v2.0
- Separate inference engine (`infer_vllm_v2.py`)
- Different GPT model architecture (`indextts/gpt/model_vllm_v2.py`)
- Currently has no acceleration advantage over v1/v1.5
- S2MEL module is the main performance bottleneck (requires 25 DiT iterations)

## Known Issues and TODOs

- **V2 Concurrency**: Only GPT model inference is parallel; S2MEL and other modules are serial
- **S2MEL Performance**: The DiT iteration process (25 steps) significantly impacts concurrency
- **v1/v1.5 OpenAI API**: May have bugs, needs fixing

## File Structure Notes

- Model checkpoints should be placed in `checkpoints/` directory
- Speaker configurations are stored in `assets/speaker.json`
- Reference audio files are typically in `assets/` directory
- The `patch_vllm.py` file contains custom vLLM modifications for TTS-specific optimizations