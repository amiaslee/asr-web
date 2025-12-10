import argparse
from pathlib import Path

import torch
import torchaudio
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperFeatureExtractor,
)

WHISPER_FEAT_CFG = {
    "chunk_length": 30,
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 128,
    "hop_length": 160,
    "n_fft": 400,
    "n_samples": 480000,
    "nb_max_frames": 3000,
    "padding_side": "right",
    "padding_value": 0.0,
    "processor_class": "WhisperProcessor",
    "return_attention_mask": False,
    "sampling_rate": 16000,
}


def get_audio_token_length(seconds, merge_factor=2):
    def get_T_after_cnn(L_in, dilation=1):
        for padding, kernel_size, stride in eval("[(1,3,1)] + [(1,3,2)] "):
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out

    mel_len = int(seconds * 100)
    audio_len_after_cnn = get_T_after_cnn(mel_len)
    audio_token_num = (audio_len_after_cnn - merge_factor) // merge_factor + 1
    
    # Limit max tokens for a single 30s chunk
    audio_token_num = min(audio_token_num, 1500)
    return audio_token_num

def prepare_single_chunk_input(
    chunk_wav,
    tokenizer,
    feature_extractor,
    merge_factor: int,
):
    """
    处理单个音频片段(30s)，构建 Prompt
    """
    # 提取特征
    mel = feature_extractor(
        chunk_wav.numpy(),
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding="max_length", # 保证不足30s的部分补零
    )["input_features"]
    
    # 计算实际秒数 (去除padding影响Token计算，但在Whisper特征中是padding的)
    seconds = chunk_wav.shape[1] / feature_extractor.sampling_rate
    num_tokens = get_audio_token_length(seconds, merge_factor)

    # 构建 Prompt Token
    tokens = []
    tokens += tokenizer.encode("<|user|>")
    tokens += tokenizer.encode("\n")
    
    # 音频部分
    tokens += tokenizer.encode("<|begin_of_audio|>")
    audio_offset = len(tokens)
    tokens += [0] * num_tokens # 占位符
    tokens += tokenizer.encode("<|end_of_audio|>")
    audio_len = num_tokens

    tokens += tokenizer.encode("<|user|>")
    tokens += tokenizer.encode("\nPlease transcribe this audio into text")
    tokens += tokenizer.encode("<|assistant|>")
    tokens += tokenizer.encode("\n")

    batch = {
        "input_ids": torch.tensor([tokens], dtype=torch.long),
        "audios": mel, # [1, 128, 3000]
        "audio_offsets": [[audio_offset]],
        "audio_length": [[audio_len]],
        "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
    }
    return batch


def prepare_inputs(batch: dict, device: torch.device) -> tuple[dict, int]:
    tokens = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    audios = batch["audios"].to(device)
    model_inputs = {
        "inputs": tokens,
        "attention_mask": attention_mask,
        "audios": audios.to(torch.bfloat16),
        "audio_offsets": batch["audio_offsets"],
        "audio_length": batch["audio_length"],
    }
    return model_inputs, tokens.size(1)


def transcribe(
    checkpoint_dir: Path,
    audio_path: Path,
    tokenizer_path: str,
    max_new_tokens: int,
    device: str,
):
    print(f"Loading model from {checkpoint_dir}...")
    tokenizer_source = tokenizer_path if tokenizer_path else checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    feature_extractor = WhisperFeatureExtractor(**WHISPER_FEAT_CFG)

    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # 1. 统一加载整个音频文件
    print(f"Loading audio file: {audio_path}")
    wav, sr = torchaudio.load(str(audio_path))
    wav = wav[:1, :] # 取单声道
    
    # 重采样到 16k
    if sr != feature_extractor.sampling_rate:
        print("Resampling audio...")
        resampler = torchaudio.transforms.Resample(sr, feature_extractor.sampling_rate)
        wav = resampler(wav)

    # 2. 分段处理逻辑
    chunk_size = 30 * feature_extractor.sampling_rate # 30秒对应的采样点数
    full_transcript = []
    
    print("Starting transcription stream...")
    print("---------- START ----------")
    
    # 循环切片
    for i, start in enumerate(range(0, wav.shape[1], chunk_size)):
        # 切割当前片段
        end = min(start + chunk_size, wav.shape[1])
        chunk = wav[:, start : end]
        
        # 构建当前片段的 Prompt
        batch = prepare_single_chunk_input(
            chunk,
            tokenizer,
            feature_extractor,
            merge_factor=config.merge_factor,
        )

        model_inputs, prompt_len = prepare_inputs(batch, device)

        # 推理
        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1 # 增加一点惩罚防止长文本复读
            )
        
        # 解码结果
        transcript_ids = generated[0, prompt_len:].cpu().tolist()
        text_chunk = tokenizer.decode(transcript_ids, skip_special_tokens=True).strip()
        
        # 实时打印
        if text_chunk:
            print(text_chunk, end=" ", flush=True)
            full_transcript.append(text_chunk)
            
        # 显存清理 (可选，防止显存碎片)
        del model_inputs, generated, batch
        torch.cuda.empty_cache()

    print("\n---------- END ----------")
    # 可选：返回完整文本
    # return " ".join(full_transcript)


def main():
    parser = argparse.ArgumentParser(description="Robust ASR transcription for long audio.")
    parser.add_argument(
        "--checkpoint_dir", type=str, default=str(Path(__file__).parent)
    )
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file.")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Tokenizer directory (defaults to checkpoint dir when omitted).",
    )
    # 对于分段识别，max_new_tokens 指的是每一段(30s)生成的最大长度
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    transcribe(
        checkpoint_dir=Path(args.checkpoint_dir),
        audio_path=Path(args.audio),
        tokenizer_path=args.tokenizer_path,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )


if __name__ == "__main__":
    main()
