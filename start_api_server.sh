#!/bin/bash
# IndexTTS API 服务器启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================="
echo -e "IndexTTS API 服务器启动脚本"
echo -e "==================================================${NC}"

# 1. 加载 .env 文件 (如果存在)
if [ -f .env ]; then
    echo -e "${GREEN}检测到 .env 文件，正在加载配置...${NC}"
    # 导出 .env 中的所有变量
    set -a
    source .env
    set +a
else
    echo -e "${YELLOW}未检测到 .env 文件，将使用系统环境变量或默认值。${NC}"
fi

# 2. 设置默认值 (如果环境变量中没有)
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-6006}
MODEL_DIR=${MODEL_DIR:-"checkpoints/IndexTTS-2-vLLM"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.25"}
QWENEMO_GPU_MEMORY_UTILIZATION=${QWENEMO_GPU_MEMORY_UTILIZATION:-"0.10"}
IS_FP16=${IS_FP16:-"0"}
ENABLE_QWEN_EMO=${ENABLE_QWEN_EMO:-"1"}
VERBOSE=${VERBOSE:-"0"}

# Python 启动命令
PY_CMD=(python3)

# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}警告: uv 未安装，将直接使用 python3 启动（不会自动创建/使用 uv 环境）${NC}"
else
    PY_CMD=(uv run python)
fi

# 检查依赖
echo -e "${YELLOW}检查依赖...${NC}"
"${PY_CMD[@]}" -c "import torch, vllm, fastapi" 2>/dev/null || {
    if command -v uv &> /dev/null; then
        echo -e "${RED}错误: 依赖未安装,请先运行: uv sync${NC}"
    else
        echo -e "${RED}错误: 依赖未安装,请先运行: pip install -r requirements.txt${NC}"
    fi
    exit 1
}

# 检查CUDA
"${PY_CMD[@]}" -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null

# 配置 CUDA 扩展编译所需的编译器 (GCC <= 12)
if [[ -z "${CC:-}" || -z "${CXX:-}" || -z "${CUDAHOSTCXX:-}" ]]; then
    if command -v gcc-12 >/dev/null 2>&1 && command -v g++-12 >/dev/null 2>&1; then
        export CC="${CC:-$(command -v gcc-12)}"
        export CXX="${CXX:-$(command -v g++-12)}"
        export CUDAHOSTCXX="${CUDAHOSTCXX:-$(command -v g++-12)}"
        echo -e "${GREEN}使用 GCC/G++ 12 作为 CUDA Host 编译器:${NC} $CC / $CXX"
    else
        echo -e "${YELLOW}警告: 未找到 gcc-12/g++-12，CUDA 扩展将回退到 PyTorch 实现。如需启用优化，请安装 GCC 12 并重新运行。${NC}"
    fi
else
    echo -e "${GREEN}检测到用户自定义编译器设置 (CC/CXX/CUDAHOSTCXX)，跳过自动配置。${NC}"
fi

# 构造 Python 脚本参数
PY_ARGS=("--host" "$HOST" "--port" "$PORT" "--model_dir" "$MODEL_DIR" "--gpu_memory_utilization" "$GPU_MEMORY_UTILIZATION" "--qwenemo_gpu_memory_utilization" "$QWENEMO_GPU_MEMORY_UTILIZATION")

if [[ "$IS_FP16" == "1" ]]; then
    PY_ARGS+=("--is_fp16")
    FP16_STATUS="启用"
else
    FP16_STATUS="禁用"
fi

if [[ "$ENABLE_QWEN_EMO" != "1" ]]; then
    PY_ARGS+=("--disable_qwen_emo")
    QWEN_EMO_STATUS="禁用"
else
    QWEN_EMO_STATUS="启用"
fi

if [[ "$VERBOSE" == "1" ]]; then
    PY_ARGS+=("--verbose")
    VERBOSE_STATUS="启用"
else
    VERBOSE_STATUS="禁用"
fi

echo -e "${GREEN}当前加载配置:${NC}"
echo -e "  主机: ${HOST}"
echo -e "  端口: ${PORT}"
echo -e "  模型目录: ${MODEL_DIR}"
echo -e "  FP16: ${FP16_STATUS}"
echo -e "  GPU内存占用率: ${GPU_MEMORY_UTILIZATION}"
echo -e "  Qwen情感GPU内存占用率: ${QWENEMO_GPU_MEMORY_UTILIZATION}"
echo -e "  Qwen情感模型: ${QWEN_EMO_STATUS}"
echo -e "  详细输出: ${VERBOSE_STATUS}"

echo -e "${BLUE}=================================================="
echo -e "正在启动 IndexTTS API 服务器..."
echo -e "==================================================${NC}"

# 启动API服务器
exec "${PY_CMD[@]}" api_server_v2.py "${PY_ARGS[@]}"
