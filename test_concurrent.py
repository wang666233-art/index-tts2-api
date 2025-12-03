#!/usr/bin/env python
"""
并发性能测试脚本
比较串行 vs 并发请求的总耗时
"""
import asyncio
import aiohttp
import time
import argparse
import json
from pathlib import Path


async def send_tts_request(session: aiohttp.ClientSession, url: str, spk_audio_path: str, text: str, request_id: int):
    """发送单个 TTS 请求"""
    payload = {
        "text": text,
        "spk_audio_path": spk_audio_path,
        "emo_control_method": 0,
    }
    
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.read()
                elapsed = time.perf_counter() - start
                print(f"  [请求 {request_id}] 完成，耗时: {elapsed:.2f}s, 音频大小: {len(data)} bytes")
                return elapsed, len(data)
            else:
                error = await response.text()
                print(f"  [请求 {request_id}] 失败: {response.status} - {error[:100]}")
                return None, 0
    except Exception as e:
        print(f"  [请求 {request_id}] 异常: {e}")
        return None, 0


async def test_sequential(url: str, spk_audio_path: str, texts: list):
    """串行测试"""
    print("\n=== 串行测试 ===")
    total_start = time.perf_counter()
    
    async with aiohttp.ClientSession() as session:
        results = []
        for i, text in enumerate(texts):
            result = await send_tts_request(session, url, spk_audio_path, text, i + 1)
            results.append(result)
    
    total_time = time.perf_counter() - total_start
    successful = [r for r in results if r[0] is not None]
    
    print(f"\n串行结果:")
    print(f"  总请求数: {len(texts)}")
    print(f"  成功数: {len(successful)}")
    print(f"  总耗时: {total_time:.2f}s")
    if successful:
        avg_time = sum(r[0] for r in successful) / len(successful)
        print(f"  平均单请求耗时: {avg_time:.2f}s")
    
    return total_time, results


async def test_concurrent(url: str, spk_audio_path: str, texts: list):
    """并发测试"""
    print("\n=== 并发测试 ===")
    total_start = time.perf_counter()
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_tts_request(session, url, spk_audio_path, text, i + 1)
            for i, text in enumerate(texts)
        ]
        results = await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - total_start
    successful = [r for r in results if r[0] is not None]
    
    print(f"\n并发结果:")
    print(f"  总请求数: {len(texts)}")
    print(f"  成功数: {len(successful)}")
    print(f"  总耗时: {total_time:.2f}s")
    if successful:
        avg_time = sum(r[0] for r in successful) / len(successful)
        print(f"  平均单请求耗时: {avg_time:.2f}s")
    
    return total_time, results


async def test_pipeline(url: str, spk_audio_path: str, long_text: str, use_pipeline: bool):
    """测试流水线模式 vs 串行模式"""
    mode = "流水线" if use_pipeline else "串行"
    print(f"\n=== {mode}模式测试 ===")
    
    payload = {
        "text": long_text,
        "spk_audio_path": spk_audio_path,
        "emo_control_method": 0,
        "use_pipeline": use_pipeline,
    }
    
    total_start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.read()
                elapsed = time.perf_counter() - total_start
                print(f"  完成，耗时: {elapsed:.2f}s, 音频大小: {len(data)} bytes")
                return elapsed, len(data)
            else:
                error = await response.text()
                print(f"  失败: {response.status} - {error[:200]}")
                return None, 0


async def main():
    parser = argparse.ArgumentParser(description="TTS 并发性能测试")
    parser.add_argument("--url", default="http://127.0.0.1:6006/tts_url", help="TTS API URL")
    parser.add_argument("--spk-audio", required=True, help="参考音频路径")
    parser.add_argument("-n", "--num-requests", type=int, default=3, help="请求数量")
    parser.add_argument("--text", default="这是一段测试文本，用于测试语音合成的并发性能。", help="合成文本")
    parser.add_argument("--test-pipeline", action="store_true", help="测试流水线模式")
    args = parser.parse_args()
    
    # 检查参考音频
    spk_audio_path = args.spk_audio
    if not Path(spk_audio_path).exists():
        # 尝试在当前目录查找
        print(f"警告: 参考音频路径不存在: {spk_audio_path}")
    
    print(f"测试配置:")
    print(f"  API URL: {args.url}")
    print(f"  参考音频: {spk_audio_path}")
    print(f"  请求数量: {args.num_requests}")
    print(f"  文本长度: {len(args.text)} 字符")
    
    # 预热请求
    print("\n=== 预热请求 ===")
    async with aiohttp.ClientSession() as session:
        await send_tts_request(session, args.url, spk_audio_path, "预热测试", 0)
    
    if args.test_pipeline:
        # 流水线模式测试（使用多句长文本）
        long_text = "这是第一句话，用于测试流水线处理。这是第二句话，继续测试。这是第三句话，验证并行效果。这是第四句话，观察性能提升。"
        
        print(f"\n测试长文本（约 {len(long_text)} 字符，多句）")
        
        # 串行模式
        serial_time, _ = await test_pipeline(args.url, spk_audio_path, long_text, use_pipeline=False)
        await asyncio.sleep(1)
        
        # 流水线模式
        pipeline_time, _ = await test_pipeline(args.url, spk_audio_path, long_text, use_pipeline=True)
        
        # 对比结果
        print("\n" + "=" * 50)
        print("流水线性能对比:")
        print(f"  串行模式耗时: {serial_time:.2f}s")
        print(f"  流水线模式耗时: {pipeline_time:.2f}s")
        if serial_time and pipeline_time and serial_time > 0:
            speedup = serial_time / pipeline_time
            improvement = (serial_time - pipeline_time) / serial_time * 100
            print(f"  加速比: {speedup:.2f}x")
            print(f"  性能提升: {improvement:.1f}%")
    else:
        # 并发请求测试
        texts = [args.text] * args.num_requests
        
        # 串行测试
        seq_time, seq_results = await test_sequential(args.url, spk_audio_path, texts)
        
        # 等待一下让 GPU 冷却
        await asyncio.sleep(1)
        
        # 并发测试
        con_time, con_results = await test_concurrent(args.url, spk_audio_path, texts)
        
        # 对比结果
        print("\n" + "=" * 50)
        print("并发性能对比:")
        print(f"  串行总耗时: {seq_time:.2f}s")
        print(f"  并发总耗时: {con_time:.2f}s")
        if seq_time > 0:
            speedup = seq_time / con_time
            print(f"  加速比: {speedup:.2f}x")
            if speedup > 1.1:
                print("  结论: 并发更快 ✅")
            elif speedup < 0.9:
                print("  结论: 串行更快 (可能存在资源竞争)")
            else:
                print("  结论: 差异不大")


if __name__ == "__main__":
    asyncio.run(main())
