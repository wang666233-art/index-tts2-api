#!/bin/bash

#need to set alias within container
alias python=python3
# Set default values if environment variables are not set
MODEL_DIR=${MODEL_DIR:-"checkpoints/"}
MODEL=${MODEL:-"IndexTeam/IndexTTS-1.5"}
VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-1}
DOWNLOAD_MODEL=${DOWNLOAD_MODEL:-1}
CONVERT_MODEL=${CONVERT_MODEL:-1}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8001}
IS_FP16=${IS_FP16:-2}
ENABLE_QWEN_EMO=${ENABLE_QWEN_EMO:-""}
DISABLE_QWEN_EMO=${DISABLE_QWEN_EMO:-0}
VERBOSE=${VERBOSE:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.25}
QWENEMO_GPU_MEMORY_UTILIZATION=${QWENEMO_GPU_MEMORY_UTILIZATION:-0.10}

normalize_switch() {
    case "$1" in
        1) echo 1 ;;
        2|0|"") echo 2 ;;
        *) echo 2 ;;
    esac
}

# Switch normalization: 1=enable, 2=disable
IS_FP16=$(normalize_switch "$IS_FP16")
VERBOSE=$(normalize_switch "$VERBOSE")
DOWNLOAD_MODEL=$(normalize_switch "$DOWNLOAD_MODEL")
CONVERT_MODEL=$(normalize_switch "$CONVERT_MODEL")

if [[ -z "${ENABLE_QWEN_EMO}" ]]; then
    if [[ "${DISABLE_QWEN_EMO}" == "1" || "${DISABLE_QWEN_EMO}" == "2" ]]; then
        ENABLE_QWEN_EMO_EFFECTIVE=2
    else
        ENABLE_QWEN_EMO_EFFECTIVE=1
    fi
else
    ENABLE_QWEN_EMO_EFFECTIVE=$(normalize_switch "${ENABLE_QWEN_EMO}")
fi

echo "Starting IndexTTS server..."
echo "Model directory: $MODEL_DIR"
echo "Model: $MODEL"
echo "Use ModelScope: $VLLM_USE_MODELSCOPE"
echo "Host: $HOST"
echo "Port: $PORT"
echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "QwenEmo GPU memory utilization: $QWENEMO_GPU_MEMORY_UTILIZATION"
echo "FP16 (1=enable,2=disable): $IS_FP16"
echo "Qwen emotion (1=enable,2=disable): $ENABLE_QWEN_EMO_EFFECTIVE"
echo "Verbose (1=enable,2=disable): $VERBOSE"

# Function to check if model directory exists and has required files
check_model_exists() {
    if [ ! -d "$MODEL_DIR" ]; then
        echo "Model directory $MODEL_DIR does not exist"
        return 1
    fi

    # Check for essential model files
    if [ ! -f "$MODEL_DIR/config.yaml" ] || [ ! -f "$MODEL_DIR/gpt.pth" ] || [ ! -f "$MODEL_DIR/bigvgan_generator.pth" ]; then
        echo "Essential model files not found in $MODEL_DIR"
        return 1
    fi

    # Download marker is optional for offline volume mounts.
    if [ ! -f "$MODEL_DIR/.download_complete" ]; then
        echo "Warning: download marker file missing, assuming model is provided by volume"
    fi
    
    echo "Model files found in $MODEL_DIR"
    return 0
}

# Function to check if model conversion is complete
check_conversion_complete() {
    if [ -f "$MODEL_DIR/.conversion_complete" ] || { [ -d "$MODEL_DIR/vllm" ] && [ -f "$MODEL_DIR/vllm/model.safetensors" ]; }; then
        echo "Model conversion already completed"
        return 0
    fi
    return 1
}

# Function to download model from HuggingFace
download_from_huggingface() {
    echo "Downloading model from HuggingFace: $MODEL"
    
    # Create model directory
    mkdir -p "$MODEL_DIR"
    
    # Use huggingface-cli to download the model
    if ! huggingface-cli download "$MODEL" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False; then
        echo "Error: Failed to download model from HuggingFace"
        exit 1
    fi
    
    # Create download marker file
    touch "$MODEL_DIR/.download_complete"
    echo "Download completed successfully!"
}

# Function to download model from ModelScope
download_from_modelscope() {
    echo "Downloading model from ModelScope: $MODEL"
    
    # Create model directory
    mkdir -p "$MODEL_DIR"
    
    # Use modelscope CLI to download the model
    if ! modelscope download --model "$MODEL" --local_dir "$MODEL_DIR"; then
        echo "Error: Failed to download model from ModelScope"
        exit 1
    fi
    
    # Create download marker file
    touch "$MODEL_DIR/.download_complete"
    echo "Download completed successfully!"
}

# Check if model exists and download if necessary
if [ "$DOWNLOAD_MODEL" = "1" ]; then
    if ! check_model_exists; then
        echo "Model not found, downloading..."
        
        # Download based on VLLM_USE_MODELSCOPE setting
        if [ "$VLLM_USE_MODELSCOPE" = "1" ]; then
            download_from_modelscope
        else
            download_from_huggingface
        fi
        
        # Verify download
        if ! check_model_exists; then
            echo "Error: Model download failed or files are missing"
            exit 1
        fi
    else
        echo "Model already exists, skipping download"
    fi
else
    echo "Model download disabled (DOWNLOAD_MODEL=$DOWNLOAD_MODEL)"
    if ! check_model_exists; then
        echo "Error: Model not found and download is disabled"
        exit 1
    fi
fi

# Convert model format if requested
if [ "$CONVERT_MODEL" = "1" ]; then
    if ! check_conversion_complete; then
        echo "Converting model format..."
        # Run conversion and capture the exit code
        bash convert_hf_format.sh "$MODEL_DIR"
        conversion_exit_code=$?
        
        # Check if conversion was successful by verifying the vllm directory exists
        if [ $conversion_exit_code -eq 0 ] && [ -d "$MODEL_DIR/vllm" ] && [ -f "$MODEL_DIR/vllm/model.safetensors" ]; then
            # Create conversion marker file on success
            touch "$MODEL_DIR/.conversion_complete"
            echo "Model conversion completed successfully"
        else
            echo "Error: Model conversion failed (exit code: $conversion_exit_code)"
            exit 1
        fi
    else
        echo "Model conversion already completed, skipping"
    fi
else
    echo "Model conversion disabled (CONVERT_MODEL=$CONVERT_MODEL)"
fi

# Start the API server
echo "Starting IndexTTS API server on $HOST:$PORT..."

CMD=(python3 api_server_v2.py
    --host "$HOST"
    --port "$PORT"
    --model_dir "$MODEL_DIR"
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
    --qwenemo_gpu_memory_utilization "$QWENEMO_GPU_MEMORY_UTILIZATION"
)

if [[ "$IS_FP16" == "1" ]]; then
    CMD+=(--is_fp16)
fi

if [[ "$ENABLE_QWEN_EMO_EFFECTIVE" == "2" ]]; then
    CMD+=(--disable_qwen_emo)
fi

if [[ "$VERBOSE" == "1" ]]; then
    CMD+=(--verbose)
fi

exec "${CMD[@]}"
