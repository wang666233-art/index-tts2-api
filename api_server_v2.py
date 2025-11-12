import os
import asyncio
import io
import traceback
import tempfile
from fastapi import FastAPI, Request, Response, File, UploadFile, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import json
import time
import soundfile as sf
from typing import List, Optional, Union

from loguru import logger
logger.add("logs/api_server_v2.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

from indextts.infer_vllm_v2 import IndexTTS2

tts = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS2(
        model_dir=args.model_dir,
        is_fp16=args.is_fp16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        qwenemo_gpu_memory_utilization=args.qwenemo_gpu_memory_utilization,
        enable_qwen_emo=not args.disable_qwen_emo,
    )
    yield


app = FastAPI(lifespan=lifespan)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if tts is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "TTS model not initialized"
            }
        )
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Service is running",
            "timestamp": time.time()
        }
    )


@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        emo_control_method = data.get("emo_control_method", 0)
        text = data["text"]
        spk_audio_path = data["spk_audio_path"]
        emo_ref_path = data.get("emo_ref_path", None)
        emo_weight = data.get("emo_weight", 1.0)
        emo_vec = data.get("emo_vec", [0] * 8)
        emo_text = data.get("emo_text", None)
        emo_random = data.get("emo_random", False)
        max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 120)

        global tts
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        if emo_control_method == 1:
            emo_weight = emo_weight
        if emo_control_method == 2:
            vec = emo_vec
            vec_sum = sum(vec)
            if vec_sum > 1.5:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "error": "情感向量之和不能超过1.5，请调整后重试。"
                    }
                )
        else:
            vec = None

        # logger.info(f"Emo control mode:{emo_control_method}, vec:{vec}")
        sr, wav = await tts.infer(spk_audio_prompt=spk_audio_path, text=text,
                        output_path=None,
                        emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                        emo_vector=vec,
                        use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                        max_text_tokens_per_sentence=int(max_text_tokens_per_sentence))
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


@app.post("/tts_upload", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_upload(
    spk_audio: UploadFile = File(...),
    text: str = Form(...),
    emo_control_method: int = Form(0),
    emo_weight: float = Form(1.0),
    emo_audio: Optional[UploadFile] = File(None),
    emo_text: Optional[str] = Form(None),
    emo_random: bool = Form(False),
    max_text_tokens_per_sentence: int = Form(120)
):
    """TTS API with file upload support"""
    try:
        # 保存上传的参考音频文件
        spk_audio_path = None
        emo_ref_path = None

        # 处理主参考音频
        if spk_audio:
            temp_dir = "/tmp/tts_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            spk_audio_path = os.path.join(temp_dir, f"spk_{int(time.time() * 1000)}.wav")

            with open(spk_audio_path, "wb") as f:
                content = await spk_audio.read()
                f.write(content)
            logger.info(f"Saved speaker reference audio: {spk_audio_path}")

        # 处理情感参考音频（如果提供）
        if emo_audio:
            emo_ref_path = os.path.join(temp_dir, f"emo_{int(time.time() * 1000)}.wav")
            with open(emo_ref_path, "wb") as f:
                content = await emo_audio.read()
                f.write(content)
            logger.info(f"Saved emotion reference audio: {emo_ref_path}")

        global tts
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        if emo_control_method == 1:
            emo_weight = emo_weight
        if emo_control_method == 2:
            # 注意：文件上传方式暂不支持向量控制
            logger.warning("Vector control method not supported with file upload, switching to method 0")
            emo_control_method = 0
            emo_weight = 1.0
            emo_ref_path = None
        else:
            emo_ref_path = emo_ref_path if emo_ref_path else None

        # 调用TTS推理
        sr, wav = await tts.infer(
            spk_audio_prompt=spk_audio_path,
            text=text,
            output_path=None,
            emo_audio_prompt=emo_ref_path,
            emo_alpha=emo_weight,
            emo_vector=None,
            use_emo_text=(emo_control_method==3),
            emo_text=emo_text,
            use_random=emo_random,
            max_text_tokens_per_sentence=int(max_text_tokens_per_sentence)
        )

        # 清理临时文件
        try:
            if spk_audio_path and os.path.exists(spk_audio_path):
                os.remove(spk_audio_path)
            if emo_ref_path and os.path.exists(emo_ref_path):
                os.remove(emo_ref_path)
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temp files: {cleanup_error}")

        # 返回音频数据
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))

        # 清理临时文件（即使出错）
        try:
            if spk_audio_path and os.path.exists(spk_audio_path):
                os.remove(spk_audio_path)
            if emo_ref_path and os.path.exists(emo_ref_path):
                os.remove(emo_ref_path)
        except:
            pass

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model_dir", type=str, default="checkpoints/IndexTTS-2-vLLM", help="Model checkpoints directory")
    parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    parser.add_argument("--qwenemo_gpu_memory_utilization", type=float, default=0.10)
    parser.add_argument("--disable_qwen_emo", action="store_true", default=False, help="Disable Qwen emotion model to save GPU memory")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
    args = parser.parse_args()
    
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    uvicorn.run(app=app, host=args.host, port=args.port)