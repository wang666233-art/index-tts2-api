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

<<<<<<< HEAD
# 检查 uv 环境
if ! command -v uv &> /dev/null; then
    echo -e "${RED}错误: 未安装 uv，请参考 https://docs.astral.sh/uv/ 进行安装${NC}"
    exit 1
fi

# 同步依赖
echo -e "${YELLOW}同步依赖...${NC}"
if ! uv sync --frozen; then
    echo -e "${RED}错误: uv 同步依赖失败，请检查 pyproject.toml 与 uv.lock${NC}"
=======
# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo -e "${RED}错误: uv 未安装${NC}"
    echo -e "${YELLOW}请安装 uv: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
>>>>>>> 49a74584cb63df32df3be1860f6603736c770b4f
    exit 1
fi

# 检查依赖
echo -e "${YELLOW}检查依赖...${NC}"
<<<<<<< HEAD
uv run -- python -c "import torch, vllm, fastapi" 2>/dev/null || {
    echo -e "${RED}错误: 依赖检查失败，请运行: uv sync${NC}"
=======
uv run python -c "import torch, vllm, fastapi" 2>/dev/null || {
    echo -e "${RED}错误: 依赖未安装,请先运行: uv sync${NC}"
>>>>>>> 49a74584cb63df32df3be1860f6603736c770b4f
    exit 1
}

# 检查CUDA
<<<<<<< HEAD
uv run -- python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null

# 自动检测 CUDA Compute Capability 以加速自定义算子编译
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    echo -e "${YELLOW}检测 GPU 架构以设置 TORCH_CUDA_ARCH_LIST...${NC}"
    if CUDA_ARCH=$(uv run -- python - <<'PY'
import sys
try:
    import torch
except Exception:
    sys.exit(2)

if not torch.cuda.is_available():
    sys.exit(3)

major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
    ); then
        CUDA_ARCH=${CUDA_ARCH//$'\r'/}
        export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}"
        echo -e "${GREEN}已设置 TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}${NC}"
    else
        echo -e "${YELLOW}未能自动检测 GPU 架构，可手动设置 TORCH_CUDA_ARCH_LIST 以优化编译时间。${NC}"
    fi
=======
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null

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
>>>>>>> 49a74584cb63df32df3be1860f6603736c770b4f
fi

# 默认参数
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-6006}
MODEL_DIR=${MODEL_DIR:-"checkpoints/IndexTTS-2-vLLM"}

# 解析命令行参数
USE_FP16=""
DISABLE_QWEN_EMO=""
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --is_fp16)
            USE_FP16="--is_fp16"
            shift
            ;;
        --disable_qwen_emo)
            DISABLE_QWEN_EMO="--disable_qwen_emo"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --host HOST           绑定地址 (默认: 0.0.0.0)"
            echo "  --port PORT           端口 (默认: 6006)"
            echo "  --model-dir DIR       模型目录 (默认: checkpoints/IndexTTS-2-vLLM)"
            echo "  --is_fp16            使用FP16精度"
            echo "  --disable_qwen_emo    禁用Qwen情感模型"
            echo "  --verbose            详细输出"
            echo "  --help               显示帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --is_fp16 --disable_qwen_emo"
            echo "  $0 --host 0.0.0.0 --port 6006 --is_fp16"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}启动配置:${NC}"
echo -e "  主机: ${HOST}"
echo -e "  端口: ${PORT}"
echo -e "  模型目录: ${MODEL_DIR}"
echo -e "  FP16: ${USE_FP16:-"未启用"}"
echo -e "  禁用Qwen情感: ${DISABLE_QWEN_EMO:-"未禁用"}"
echo -e "  详细输出: ${VERBOSE:-"未启用"}"

echo -e "${BLUE}=================================================="
echo -e "启动 IndexTTS API 服务器..."
echo -e "==================================================${NC}"

# 启动API服务器
<<<<<<< HEAD
exec uv run -- python api_server_v2.py \
=======
exec uv run python api_server_v2.py \
>>>>>>> 49a74584cb63df32df3be1860f6603736c770b4f
    --host "$HOST" \
    --port "$PORT" \
    --model_dir "$MODEL_DIR" \
    $USE_FP16 \
    $DISABLE_QWEN_EMO \
    $VERBOSE
