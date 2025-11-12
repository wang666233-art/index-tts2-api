#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_simple_vllm.py - Simple vLLM Test

Test basic vLLM functionality using the same approach as the deleted tts_lrc_to_audio.py
"""

import time
import asyncio
import os
import sys
from pathlib import Path

def test_simple_vllm_init():
    """Test vLLM initialization using the original working approach."""
    print("=== Testing Simple vLLM Initialization ===")

    try:
        # Import and test like the original working code
        from indextts.infer_vllm_v2 import IndexTTS2

        print("Import successful")

        # Use the same model path as the working code
        model_dir = "checkpoints/IndexTTS-2-vLLM"

        if not os.path.exists(model_dir):
            print(f"❌ Model directory not found: {model_dir}")
            return False

        print(f"Using model directory: {model_dir}")

        # Initialize like the original working code
        start_time = time.time()

        tts = IndexTTS2(
            model_dir=model_dir,
            gpu_memory_utilization=0.85  # Use conservative setting
        )

        init_time = time.time() - start_time
        print(f"✅ vLLM initialization successful in {init_time:.2f}s")

        # Test basic functionality
        print("Testing basic inference...")

        # Create a dummy reference audio file for testing
        dummy_ref_path = "dummy_ref.wav"
        import soundfile as sf
        import numpy as np

        # Create 1 second of silence as reference
        dummy_audio = np.zeros(22050, dtype=np.float32)
        sf.write(dummy_ref_path, dummy_audio, 22050)

        # Run a simple inference
        async def test_inference():
            sr, wav = await tts.infer(
                spk_audio_prompt=dummy_ref_path,
                text="これはテストです",
                output_path=None,
                max_text_tokens_per_sentence=50
            )
            print(f"✅ Inference successful: generated {len(wav)/sr:.2f}s of audio")
            return True

        # Run the test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(test_inference())
        finally:
            loop.close()

        # Cleanup
        if os.path.exists(dummy_ref_path):
            os.remove(dummy_ref_path)

        # Test performance monitoring
        stats = tts.get_performance_stats()
        print(f"Performance stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        return result

    except Exception as e:
        print(f"❌ Simple vLLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the simple test."""
    print("Starting Simple vLLM Test")
    print("=" * 40)

    # Check if we're in the right directory
    if not os.path.exists("checkpoints/IndexTTS-2-vLLM"):
        print("❌ Not in the correct directory. Please run from index-tts-vllm/")
        return False

    # Run the test
    success = test_simple_vllm_init()

    print("\n" + "=" * 40)
    if success:
        print("🎉 vLLM is working correctly!")
    else:
        print("❌ vLLM test failed")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)