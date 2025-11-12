#!/usr/bin/env python3
"""
IndexTTS API 服务器依赖安装器
支持CUDA并确保与vLLM兼容性
"""

import os
import sys
import subprocess
import platform

def run_command(cmd, description="", check=True):
    """运行命令并处理错误"""
    print(f"\n{'='*50}")
    print(f"执行: {description}")
    print(f"命令: {cmd}")
    print('='*50)

    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True,
                              encoding='utf-8', errors='ignore')
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        if e.stderr:
            print(f"错误详情: {e.stderr}")
        if check:
            print("安装失败，请检查上面的错误信息")
            sys.exit(1)
        return e

def check_nvidia_gpu():
    """检查是否有NVIDIA GPU"""
    print("检查NVIDIA GPU支持...")
    result = run_command("nvidia-smi", check=False)
    if result.returncode == 0:
        print("✓ 检测到NVIDIA GPU")
        return True
    else:
        print("❌ 未检测到NVIDIA GPU或nvidia-smi不可用")
        return False

def ask_cuda_choice():
    """询问用户是否安装CUDA版本"""
    print("\n" + "=" * 60)
    print("PyTorch 安装选择")
    print("=" * 60)
    print("1. CUDA版本 (推荐，需要NVIDIA GPU)")
    print("2. CPU版本 (速度较慢，但兼容性更好)")

    while True:
        choice = input("\n请选择 (1 或 2): ").strip()
        if choice in ['1', '2']:
            return choice == '1'
        print("无效选择，请输入 1 或 2")

def check_torch():
    """检查PyTorch是否正确安装"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    except ImportError:
        print("❌ PyTorch 未安装")
        return False

def main():
    print("IndexTTS API 服务器依赖安装器")
    print("=" * 60)

    # 检查GPU并询问用户选择
    has_cuda_gpu = check_nvidia_gpu()

    if has_cuda_gpu:
        use_cuda = ask_cuda_choice()
    else:
        print("未检测到GPU，将使用CPU版本")
        use_cuda = False

    # 检查pip版本
    print("确保pip是最新的...")
    run_command("python -m pip install --upgrade pip", "升级pip")

    # 安装PyTorch
    if use_cuda:
        print("安装CUDA版本的PyTorch...")
        run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
                   "安装PyTorch CUDA 12.1版本")
    else:
        print("安装CPU版本的PyTorch...")
        run_command("pip install torch torchvision torchaudio", "安装PyTorch CPU版本")

    # 安装基础依赖
    print("安装API服务器依赖...")
    run_command("pip install -r requirements.txt", "安装基础依赖")

    # 验证安装
    print("\n" + "=" * 60)
    print("验证安装...")
    print("=" * 60)

    if check_torch():
        print("✅ PyTorch 安装成功")
    else:
        print("❌ PyTorch 安装失败")
        sys.exit(1)

    # 测试vLLM
    try:
        import vllm
        print(f"✅ vLLM 安装成功，版本: {vllm.__version__}")
    except ImportError as e:
        print(f"❌ vLLM 安装失败: {e}")
        sys.exit(1)

    # 测试FastAPI
    try:
        import fastapi
        print(f"✅ FastAPI 安装成功，版本: {fastapi.__version__}")
    except ImportError as e:
        print(f"❌ FastAPI 安装失败: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🎉 IndexTTS API 服务器依赖安装完成！")
    print("=" * 60)
    print("\n启动API服务器:")
    print("python api_server_v2.py --is_fp16 --disable_qwen_emo")
    print("\n如果使用CUDA，建议添加 --host 0.0.0.0 以允许远程访问:")
    print("python api_server_v2.py --is_fp16 --disable_qwen_emo --host 0.0.0.0")

if __name__ == "__main__":
    main()