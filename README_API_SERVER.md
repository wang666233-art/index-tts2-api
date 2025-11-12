# IndexTTS API 服务器

独立的 TTS API 服务器，支持分布式部署，可与 kikoeru-translate 分离运行。

## 功能特点

- ✅ **独立部署**: 可在专门的 GPU 服务器上运行
- ✅ **版本隔离**: 避免与 kikoeru-translate 的依赖冲突
- ✅ **CUDA 支持**: 自动检测并支持 NVIDIA GPU
- ✅ **文件上传**: 支持通过 HTTP 上传参考音频
- ✅ **双端点**: 支持文件路径和文件上传两种方式
- ✅ **自动清理**: 临时文件自动管理

## 快速开始

### 1. 创建虚拟环境

```bash
# 创建新的虚拟环境
python -m venv tts-api-env

# 激活虚拟环境
# Linux/macOS:
source tts-api-env/bin/activate
# Windows:
tts-api-env\Scripts\activate
```

### 2. 安装依赖

```bash
# 进入 API 服务器目录
cd index-tts-vllm

# 运行自动安装脚本
python install_api_server.py

# 或手动安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. 下载模型

```bash
# 下载 IndexTTS v2.0 模型
modelscope download --model kusuriuri/IndexTTS-2-vLLM --local_dir ./checkpoints/IndexTTS-2-vLLM
```

### 4. 启动服务器

```bash
# 使用启动脚本（推荐）
./start_api_server.sh --is_fp16 --disable_qwen_emo

# 或直接运行
python api_server_v2.py --is_fp16 --disable_qwen_emo --host 0.0.0.0
```

## API 端点

### 1. `/tts_url` - 文件路径方式
使用本地文件路径，需要共享文件系统。

**请求**:
```json
{
    "text": "要合成的文本",
    "spk_audio_path": "/path/to/reference.wav",
    "emo_control_method": 1,
    "emo_weight": 1.0
}
```

### 2. `/tts_upload` - 文件上传方式
通过 HTTP 上传参考音频，支持分布式部署。

**请求**: `multipart/form-data`
- `spk_audio`: 参考音频文件
- `text`: 合成文本
- `emo_control_method`: 情感控制方法
- `emo_weight`: 情感权重
- `emo_audio`: 情感参考音频（可选）

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | 0.0.0.0 | 绑定地址 |
| `--port` | 6006 | 端口号 |
| `--model_dir` | checkpoints/IndexTTS-2-vLLM | 模型目录 |
| `--is_fp16` | False | 使用FP16精度（推荐启用） |
| `--disable_qwen_emo` | False | 禁用Qwen情感模型（节省内存） |
| `--verbose` | False | 详细输出 |

## 使用示例

### 本地测试

```bash
# 启动服务器
./start_api_server.sh --is_fp16 --disable_qwen_emo

# 测试API
curl -X POST "http://localhost:6006/tts_url" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "你好，这是一个测试。",
       "spk_audio_path": "/path/to/reference.wav"
     }'
```

### 分布式部署

```bash
# 在 GPU 服务器上启动 API 服务器
./start_api_server.sh --is_fp16 --disable_qwen_emo --host 0.0.0.0

# 在 kikoeru-translate 中配置 API 地址
# 修改 kikoeru_worker.py 中的 api_url
api_url="http://gpu-server-ip:6006/tts_url"
```

## 与 kikoeru-translate 集成

1. 确保 API 服务器正在运行
2. 修改 kikoeru_worker.py 中的 API 地址
3. 启用文件上传方式（推荐用于分布式部署）

```python
# kikoeru_worker.py 中的配置
chinese_audio, sr, tmp_dir = synthesize_chinese(
    subtitle_path=zh_lrc_path,
    spk_ref=best_reference,
    final_sr=common.getTTSSamplingRate(),
    api_url="http://api-server:6006/tts_url",  # 指向API服务器
    use_file_upload=True  # 启用文件上传
)
```

## 故障排除

### 1. CUDA 相关问题
```bash
# 检查 CUDA 版本
nvidia-smi

# 检查 PyTorch CUDA 支持
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 模型路径问题
确保模型目录存在且包含必要的文件：
```
checkpoints/IndexTTS-2-vLLM/
├── gpt/
├── s2mel/
└── bigvgan/
```

### 3. 端口占用
```bash
# 检查端口占用
netstat -tlnp | grep 6006

# 更换端口
./start_api_server.sh --port 6007
```

## 性能优化

1. **启用 FP16**: 使用 `--is_fp16` 参数，显著提升性能
2. **禁用情感模型**: 使用 `--disable_qwen_emo` 节省 GPU 内存
3. **GPU 内存管理**: 调整 `gpu_memory_utilization` 参数

## 日志

API 服务器日志保存在 `logs/api_server_v2.log`，包含详细的错误信息和调试信息。