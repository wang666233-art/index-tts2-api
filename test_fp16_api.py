#!/usr/bin/env python3
"""
FP16 方案 A 测试脚本
测试启用 FP16 后 API 服务器是否正常工作
"""

import requests
import time
import os
import argparse

API_BASE = "http://127.0.0.1:6006"

def test_health():
    """测试服务器健康状态"""
    print("=" * 60)
    print("测试 1: 服务器健康检查")
    print("=" * 60)
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=10)
        if resp.status_code == 200:
            print(f"  ✓ 服务器正常运行")
            print(f"    响应: {resp.json()}")
            return True
        else:
            print(f"  ✗ 服务器响应异常: {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  ✗ 无法连接到服务器 {API_BASE}")
        print(f"    请确保服务器已启动: ./start_api_server.sh --is_fp16 --disable_qwen_emo")
        return False
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        return False

def test_tts_simple(ref_audio: str, output_path: str = "test_output_fp16.wav"):
    """测试简单 TTS 合成"""
    print("\n" + "=" * 60)
    print("测试 2: 简单 TTS 合成")
    print("=" * 60)
    
    text = "你好，这是一个测试。"
    
    # 转换为绝对路径
    ref_audio_abs = os.path.abspath(ref_audio)
    
    print(f"  参考音频: {ref_audio_abs}")
    print(f"  合成文本: {text}")
    print(f"  输出路径: {output_path}")
    
    try:
        start_time = time.time()
        resp = requests.post(
            f"{API_BASE}/tts_url",
            json={
                "text": text,
                "spk_audio_path": ref_audio_abs,
            },
            timeout=120
        )
        elapsed = time.time() - start_time
        
        if resp.status_code == 200:
            # 保存音频
            with open(output_path, "wb") as f:
                f.write(resp.content)
            
            file_size = os.path.getsize(output_path)
            print(f"  ✓ 合成成功!")
            print(f"    耗时: {elapsed:.2f} 秒")
            print(f"    文件大小: {file_size / 1024:.1f} KB")
            print(f"    输出文件: {output_path}")
            return True
        else:
            print(f"  ✗ 合成失败: {resp.status_code}")
            # 显示完整错误信息
            try:
                error_data = resp.json()
                print(f"    错误详情:\n{error_data.get('error', resp.text)}")
            except:
                print(f"    响应: {resp.text}")
            return False
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tts_long(ref_audio: str, output_path: str = "test_output_fp16_long.wav"):
    """测试长文本 TTS 合成"""
    print("\n" + "=" * 60)
    print("测试 3: 长文本 TTS 合成")
    print("=" * 60)
    
    text = """这是一段较长的测试文本，用于验证 FP16 推理在处理多句话时是否稳定。
语音合成系统需要将文本转换为自然流畅的语音，这涉及到多个深度学习模型的协同工作。
希望这个测试能够顺利通过，证明我们的优化是有效的。"""
    
    # 转换为绝对路径
    ref_audio_abs = os.path.abspath(ref_audio)
    
    print(f"  参考音频: {ref_audio_abs}")
    print(f"  合成文本: {text[:50]}...")
    print(f"  文本长度: {len(text)} 字符")
    print(f"  输出路径: {output_path}")
    
    try:
        start_time = time.time()
        resp = requests.post(
            f"{API_BASE}/tts_url",
            json={
                "text": text,
                "spk_audio_path": ref_audio_abs,
            },
            timeout=180
        )
        elapsed = time.time() - start_time
        
        if resp.status_code == 200:
            # 保存音频
            with open(output_path, "wb") as f:
                f.write(resp.content)
            
            file_size = os.path.getsize(output_path)
            print(f"  ✓ 合成成功!")
            print(f"    耗时: {elapsed:.2f} 秒")
            print(f"    文件大小: {file_size / 1024:.1f} KB")
            print(f"    输出文件: {output_path}")
            return True
        else:
            print(f"  ✗ 合成失败: {resp.status_code}")
            print(f"    响应: {resp.text[:500]}")
            return False
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="FP16 方案 A 测试脚本")
    parser.add_argument("--ref_audio", type=str, default="./ref_SPEAKER_02_1764743184.wav",
                        help="参考音频路径")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:6006",
                        help="API 服务器地址")
    args = parser.parse_args()
    
    global API_BASE
    API_BASE = args.api_url
    
    print("=" * 60)
    print("IndexTTS FP16 方案 A 测试")
    print(f"API 地址: {API_BASE}")
    print(f"参考音频: {args.ref_audio}")
    print("=" * 60)
    
    # 检查参考音频是否存在
    if not os.path.exists(args.ref_audio):
        print(f"\n❌ 参考音频不存在: {args.ref_audio}")
        print("请指定有效的参考音频: --ref_audio <路径>")
        return
    
    results = {}
    
    # 测试 1: 健康检查
    results["health"] = test_health()
    if not results["health"]:
        print("\n❌ 服务器未运行，测试终止")
        return
    
    # 测试 2: 简单合成
    results["simple"] = test_tts_simple(args.ref_audio)
    
    # 测试 3: 长文本合成
    results["long"] = test_tts_long(args.ref_audio)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有测试通过! FP16 方案 A 工作正常")
    else:
        print("\n⚠️ 部分测试失败，请检查日志")

if __name__ == "__main__":
    main()
