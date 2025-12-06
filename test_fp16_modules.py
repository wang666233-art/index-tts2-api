#!/usr/bin/env python3
"""
FP16 模块兼容性测试脚本
逐个测试各模块的 FP16 支持情况
"""

import os
import sys
import torch
import torchaudio
import librosa
from loguru import logger

# 设置设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "checkpoints/IndexTTS-2-vLLM"

def test_semantic_model_fp16():
    """测试 semantic_model (Wav2Vec2BertModel) 的 FP16 支持"""
    print("\n" + "="*60)
    print("测试 1: semantic_model (Wav2Vec2BertModel)")
    print("="*60)
    
    from transformers import SeamlessM4TFeatureExtractor
    from indextts.utils.maskgct_utils import build_semantic_model
    
    try:
        # 加载模型
        extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            os.path.join(MODEL_DIR, "w2v-bert-2.0")
        )
        
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(os.path.join(MODEL_DIR, "config.yaml"))
        
        semantic_model, semantic_mean, semantic_std = build_semantic_model(
            os.path.join(MODEL_DIR, cfg.w2v_stat),
            os.path.join(MODEL_DIR, "w2v-bert-2.0")
        )
        semantic_model = semantic_model.to(DEVICE)
        semantic_mean = semantic_mean.to(DEVICE)
        semantic_std = semantic_std.to(DEVICE)
        
        # 准备测试数据
        test_audio = torch.randn(1, 16000).numpy()[0]  # 1秒音频
        inputs = extract_features(test_audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)
        
        # 测试 FP32
        print("\n[FP32 测试]")
        semantic_model.eval()
        with torch.no_grad():
            output = semantic_model(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            feat = output.hidden_states[17]
            feat = (feat - semantic_mean) / semantic_std
        print(f"  ✓ FP32 成功, 输出形状: {feat.shape}, 类型: {feat.dtype}")
        
        # 测试 FP16
        print("\n[FP16 测试]")
        semantic_model.half()
        semantic_mean_fp16 = semantic_mean.half()
        semantic_std_fp16 = semantic_std.half()
        input_features_fp16 = input_features.half()
        
        with torch.no_grad():
            output = semantic_model(
                input_features=input_features_fp16,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            feat = output.hidden_states[17]
            feat = (feat - semantic_mean_fp16) / semantic_std_fp16
        print(f"  ✓ FP16 成功, 输出形状: {feat.shape}, 类型: {feat.dtype}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_semantic_codec_fp16():
    """测试 semantic_codec 的 FP16 支持"""
    print("\n" + "="*60)
    print("测试 2: semantic_codec")
    print("="*60)
    
    from indextts.utils.maskgct_utils import build_semantic_codec
    import safetensors.torch
    from omegaconf import OmegaConf
    
    try:
        cfg = OmegaConf.load(os.path.join(MODEL_DIR, "config.yaml"))
        semantic_codec = build_semantic_codec(cfg.semantic_codec)
        semantic_code_ckpt = os.path.join(MODEL_DIR, "semantic_codec/model.safetensors")
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        semantic_codec = semantic_codec.to(DEVICE)
        
        # 准备测试数据 (模拟 semantic_model 的输出)
        test_input = torch.randn(1, 100, 1024).to(DEVICE)  # (B, T, C)
        
        # 测试 FP32
        print("\n[FP32 测试]")
        semantic_codec.eval()
        with torch.no_grad():
            _, codes = semantic_codec.quantize(test_input)
        print(f"  ✓ FP32 成功, codes 形状: {codes.shape}")
        
        # 测试 FP16
        print("\n[FP16 测试]")
        semantic_codec.half()
        test_input_fp16 = test_input.half()
        
        with torch.no_grad():
            _, codes = semantic_codec.quantize(test_input_fp16)
        print(f"  ✓ FP16 成功, codes 形状: {codes.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpt_fp16():
    """测试 GPT (UnifiedVoice) 内部组件的 FP16 支持"""
    print("\n" + "="*60)
    print("测试 3: GPT (UnifiedVoice) - 内部组件测试")
    print("="*60)
    
    from omegaconf import OmegaConf
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint
    
    try:
        cfg = OmegaConf.load(os.path.join(MODEL_DIR, "config.yaml"))
        gpt = UnifiedVoice(**cfg.gpt)
        gpt_path = os.path.join(MODEL_DIR, cfg.gpt_checkpoint)
        load_checkpoint(gpt, gpt_path)
        gpt = gpt.to(DEVICE)
        gpt.eval()
        
        # 测试内部 GPT2 模型（这是 FP16 最可能出问题的部分）
        print("\n[测试内部 GPT2 模型]")
        
        # 构造 embedding 输入
        batch_size = 1
        seq_len = 50
        model_dim = cfg.gpt.model_dim  # 1280
        
        # FP32 测试
        test_emb = torch.randn(batch_size, seq_len, model_dim).to(DEVICE)
        with torch.no_grad():
            out = gpt.gpt(inputs_embeds=test_emb, return_dict=True)
        print(f"  ✓ FP32 GPT2 成功, 输出形状: {out.last_hidden_state.shape}")
        
        # FP16 测试
        print("\n[FP16 测试 - GPT2]")
        gpt.half()
        test_emb_fp16 = test_emb.half()
        
        with torch.no_grad():
            out = gpt.gpt(inputs_embeds=test_emb_fp16, return_dict=True)
        print(f"  ✓ FP16 GPT2 成功, 输出形状: {out.last_hidden_state.shape}, 类型: {out.last_hidden_state.dtype}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_s2mel_fp16():
    """测试 s2mel 的 FP16 支持"""
    print("\n" + "="*60)
    print("测试 4: s2mel (CFM Diffusion)")
    print("="*60)
    
    from omegaconf import OmegaConf
    from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
    
    try:
        cfg = OmegaConf.load(os.path.join(MODEL_DIR, "config.yaml"))
        s2mel_path = os.path.join(MODEL_DIR, cfg.s2mel_checkpoint)
        s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel, None, s2mel_path,
            load_only_params=True, ignore_modules=[], is_distributed=False,
        )
        s2mel = s2mel.to(DEVICE)
        s2mel.eval()
        
        # 准备测试数据 (GPT model_dim=1280)
        latent = torch.randn(1, 50, 1280).to(DEVICE)  # GPT 输出
        
        # 测试 FP32 - gpt_layer
        print("\n[FP32 测试 - gpt_layer]")
        with torch.no_grad():
            out = s2mel.models['gpt_layer'](latent)
        print(f"  ✓ FP32 gpt_layer 成功, 输出形状: {out.shape}")
        
        # 测试 FP16 - gpt_layer
        print("\n[FP16 测试 - gpt_layer]")
        s2mel.half()
        latent_fp16 = latent.half()
        
        with torch.no_grad():
            out = s2mel.models['gpt_layer'](latent_fp16)
        print(f"  ✓ FP16 gpt_layer 成功, 输出形状: {out.shape}, 类型: {out.dtype}")
        
        # CFM (DiT) 测试
        print("\n[CFM DiT 测试]")
        try:
            # 重新加载模型
            s2mel_fp32 = MyModel(cfg.s2mel, use_gpt_latent=True)
            s2mel_fp32, _, _, _ = load_checkpoint2(
                s2mel_fp32, None, s2mel_path,
                load_only_params=True, ignore_modules=[], is_distributed=False,
            )
            s2mel_fp32 = s2mel_fp32.to(DEVICE)
            s2mel_fp32.eval()
            s2mel_fp32.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
            
            # 测试 DiT 的 x_embedder (这是输入投影层)
            x_embedder = s2mel_fp32.models['cfm'].estimator.x_embedder
            in_channels = s2mel_fp32.models['cfm'].estimator.in_channels  # 80
            test_x = torch.randn(1, 100, in_channels).to(DEVICE)
            
            # FP32 测试
            with torch.no_grad():
                out = x_embedder(test_x)
            print(f"  ✓ FP32 DiT x_embedder 成功, 输出形状: {out.shape}")
            
            # FP16 测试
            s2mel_fp32.half()
            test_x_fp16 = test_x.half()
            
            with torch.no_grad():
                out = x_embedder(test_x_fp16)
            print(f"  ✓ FP16 DiT x_embedder 成功, 输出形状: {out.shape}, 类型: {out.dtype}")
            
            # 测试 TimestepEmbedder (这是可能出问题的地方)
            print("\n[CFM TimestepEmbedder 测试]")
            t_embedder = s2mel_fp32.models['cfm'].estimator.t_embedder
            test_t = torch.rand(1).to(DEVICE)
            
            # FP16 测试
            with torch.no_grad():
                t_emb = t_embedder(test_t)
            print(f"  ✓ FP16 TimestepEmbedder 成功, 输出形状: {t_emb.shape}, 类型: {t_emb.dtype}")
            
        except Exception as e:
            print(f"  ✗ CFM 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_campplus_fp16():
    """测试 CAMPPlus 的 FP16 支持"""
    print("\n" + "="*60)
    print("测试 5: CAMPPlus (说话人编码器)")
    print("="*60)
    
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
    
    try:
        campplus_ckpt_path = os.path.join(MODEL_DIR, "campplus/campplus_cn_common.bin")
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        campplus_model = campplus_model.to(DEVICE)
        campplus_model.eval()
        
        # 准备测试数据 (80-dim fbank features)
        test_feat = torch.randn(1, 100, 80).to(DEVICE)  # (B, T, 80)
        
        # 测试 FP32
        print("\n[FP32 测试]")
        with torch.no_grad():
            style = campplus_model(test_feat)
        print(f"  ✓ FP32 成功, 输出形状: {style.shape}")
        
        # 测试 FP16
        print("\n[FP16 测试]")
        campplus_model.half()
        test_feat_fp16 = test_feat.half()
        
        with torch.no_grad():
            style = campplus_model(test_feat_fp16)
        print(f"  ✓ FP16 成功, 输出形状: {style.shape}, 类型: {style.dtype}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bigvgan_fp16():
    """测试 BigVGAN 的 FP16 支持"""
    print("\n" + "="*60)
    print("测试 6: BigVGAN (声码器)")
    print("="*60)
    
    from indextts.s2mel.modules.bigvgan import bigvgan
    
    try:
        # 使用 from_pretrained 加载
        model = bigvgan.BigVGAN.from_pretrained(
            os.path.join(MODEL_DIR, "bigvgan")
        )
        model = model.to(DEVICE)
        model.remove_weight_norm()
        model.eval()
        
        # 准备测试数据 (mel spectrogram) - BigVGAN 需要 80 通道
        test_mel = torch.randn(1, 80, 200).to(DEVICE)  # (B, n_mels=80, T)
        
        # 测试 FP32
        print("\n[FP32 测试]")
        with torch.no_grad():
            wav = model(test_mel)
        print(f"  ✓ FP32 成功, 输出形状: {wav.shape}")
        
        # 测试 FP16
        print("\n[FP16 测试]")
        model.half()
        test_mel_fp16 = test_mel.half()
        
        with torch.no_grad():
            wav = model(test_mel_fp16)
        print(f"  ✓ FP16 成功, 输出形状: {wav.shape}, 类型: {wav.dtype}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("IndexTTS FP16 模块兼容性测试")
    print(f"设备: {DEVICE}")
    print(f"模型目录: {MODEL_DIR}")
    print("="*60)
    
    results = {}
    
    # 按依赖顺序测试
    results["semantic_model"] = test_semantic_model_fp16()
    results["semantic_codec"] = test_semantic_codec_fp16()
    results["gpt"] = test_gpt_fp16()
    results["s2mel"] = test_s2mel_fp16()
    results["campplus"] = test_campplus_fp16()
    results["bigvgan"] = test_bigvgan_fp16()
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for name, result in results.items():
        if result is True:
            status = "✓ 支持 FP16"
        elif result is False:
            status = "✗ 不支持 FP16"
        else:
            status = "⚠ 跳过"
        print(f"  {name}: {status}")
    
    print("\n建议:")
    print("  1. 对支持 FP16 的模块启用 .half()")
    print("  2. 确保输入数据也转为 half()")
    print("  3. 不支持的模块保持 float32")


if __name__ == "__main__":
    main()
