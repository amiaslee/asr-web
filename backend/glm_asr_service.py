import torch
import torchaudio
from pathlib import Path
import tempfile
import os
import subprocess
import soundfile as sf
import numpy as np
from typing import Dict, Any
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperFeatureExtractor,
)
from base_service import BaseASRService

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


class GLMASRService(BaseASRService):
    def __init__(self, model_path="zai-org/GLM-ASR-Nano-2512"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.feature_extractor = None
        self.config = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.prompt = "Please transcribe this audio into text"  # Standard prompt for transcription
        # Auto-load model on initialization
        self.load_model()

    def load_model(self):
        print(f"Loading model {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.feature_extractor = WhisperFeatureExtractor(**WHISPER_FEAT_CFG)
        
        self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        print("Model loaded successfully.")
        return True
    
    def _load_audio(self, audio_path: str):
        """Load audio file and convert to 16kHz mono - using soundfile like demo.py"""
        temp_wav_path = None
        try:
            import soundfile as sf
            import numpy as np
            
            # Check file extension
            file_ext = os.path.splitext(audio_path)[1].lower()
            
            # If not WAV or if it's a format soundfile often struggles with (like webm), convert first
            if file_ext not in ['.wav', '.flac'] or file_ext in ['.webm', '.mp4', '.m4a']:
                print(f"Converting {file_ext} file to WAV for compatibility...")
                temp_wav_path = self.convert_to_wav(audio_path)
                load_path = temp_wav_path
            else:
                load_path = audio_path
            
            # Load audio using soundfile (same as demo.py)
            audio_data, sample_rate = sf.read(load_path, dtype='float32')
            
            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import torchaudio.transforms as T
                # Convert to tensor, resample, then back to numpy
                audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
                resampler = T.Resample(sample_rate, 16000)
                audio_tensor = resampler(audio_tensor)
                audio_data = audio_tensor.squeeze().numpy()
                sample_rate = 16000
            
            return audio_data, sample_rate
        except Exception as e:
            print(f"Error loading audio: {e}")
            raise
        finally:
            # Clean up temporary file
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except OSError:
                    pass
    
    def is_loaded(self):
        return self.model is not None

    def convert_to_wav(self, input_path):
        """
        Convert audio/video file to WAV format using ffmpeg.
        Supports: mp3, mp4, m4a, aac, ogg, flac, avi, mkv, etc.
        """
        try:
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_wav.close()
            
            subprocess.run([
                'ffmpeg',
                '-i', input_path,
                '-ar', '16000',
                '-ac', '1',
                '-y',
                temp_wav.name
            ], check=True, capture_output=True, text=True)
            
            return temp_wav.name
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Failed to convert audio/video file with ffmpeg: {e.stderr}")
        except FileNotFoundError:
            raise ValueError("ffmpeg is not installed. Please install ffmpeg: brew install ffmpeg")
        except Exception as e:
            raise ValueError(f"Failed to convert audio/video file: {str(e)}")

    def prepare_single_chunk_input(self, chunk_wav):
        """
        处理单个音频片段(30s)，构建 Prompt
        """
        # 提取特征 - squeeze to convert from [1, samples] to [samples]
        mel = self.feature_extractor(
            chunk_wav.squeeze(0).numpy(),
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding="max_length",
        )["input_features"]
        
        # 计算实际秒数
        seconds = chunk_wav.shape[1] / self.feature_extractor.sampling_rate
        num_tokens = get_audio_token_length(seconds, self.config.merge_factor)

        # 构建 Prompt Token
        tokens = []
        tokens += self.tokenizer.encode("<|user|>")
        tokens += self.tokenizer.encode("\n")
        
        # 音频部分
        tokens += self.tokenizer.encode("<|begin_of_audio|>")
        audio_offset = len(tokens)
        tokens += [0] * num_tokens  # 占位符
        tokens += self.tokenizer.encode("<|end_of_audio|>")
        audio_len = num_tokens

        tokens += self.tokenizer.encode("<|user|>")
        tokens += self.tokenizer.encode("\nPlease transcribe this audio into text")
        tokens += self.tokenizer.encode("<|assistant|>")
        tokens += self.tokenizer.encode("\n")

        batch = {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "audios": mel,
            "audio_offsets": [[audio_offset]],
            "audio_length": [[audio_len]],
            "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
        }
        return batch

    def prepare_inputs(self, batch):
        """准备模型输入"""
        tokens = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        audios = batch["audios"].to(self.device)
        
        model_inputs = {
            "inputs": tokens,
            "attention_mask": attention_mask,
            "audios": audios.to(torch.bfloat16),
            "audio_offsets": batch["audio_offsets"],
            "audio_length": batch["audio_length"],
        }
        return model_inputs, tokens.size(1)

    def transcribe(self, audio_path: str, max_tokens: int = 128, progress_callback=None):
        """Transcribe audio file with optional progress callback
        
        Args:
            audio_path: Path to audio file
            max_tokens: Maximum tokens to generate per chunk
            progress_callback: Optional callable(progress: int, status: str) for progress updates
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if progress_callback:
            progress_callback(5, "Loading audio file")
        
        print(f"Loading audio file: {audio_path}")
        audio, sample_rate = self._load_audio(audio_path)
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        
        if progress_callback:
            progress_callback(15, f"Audio loaded: {audio_tensor.shape[1]/sample_rate:.2f}s")
        
        print(f"Audio loaded: shape={audio_tensor.shape}, sample_rate={sample_rate}")
        
        # Split audio into chunks
        chunk_duration_sec = 30
        chunk_samples = chunk_duration_sec * sample_rate
        total_samples = audio_tensor.shape[1]
        
        chunks = []
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunks.append(audio_tensor[:, start:end])
        
        total_chunks = len(chunks)
        if progress_callback:
            progress_callback(20, f"Processing {total_chunks} chunks")
        
        print(f"Starting transcription... Total duration: {total_samples/sample_rate:.2f}s")
        
        all_transcriptions = []
        
        for i, chunk in enumerate(chunks):
            # Calculate progress for this chunk (20% to 85% range)
            chunk_progress_start = 20 + (i * 65) // total_chunks
            chunk_progress_end = 20 + ((i + 1) * 65) // total_chunks
            
            if progress_callback:
                progress_callback(chunk_progress_start, f"Processing chunk {i+1}/{total_chunks}")
            
            duration = chunk.shape[1] / sample_rate
            print(f"Processing chunk {i+1}: duration={duration:.2f}s, shape={chunk.shape}")
            
            # Prepare input
            if progress_callback:
                progress_callback(chunk_progress_start + 2, f"Preparing input for chunk {i+1}")
            
            print(f"  Preparing input for chunk {i+1}...")
            batch = self.prepare_single_chunk_input(chunk)
            
            print(f"  Batch prepared: input_ids shape={batch['input_ids'].shape}")
            
            model_inputs, prompt_len = self.prepare_inputs(batch)
            
            if progress_callback:
                progress_callback(chunk_progress_start + 5, f"Model inference for chunk {i+1}")
            
            # Generate
            print(f"  Model inputs prepared, prompt_len={prompt_len}")
            print(f"  Generating with max_new_tokens={max_tokens}...")
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    repetition_penalty=1.1
                )
            
            if progress_callback:
                progress_callback(chunk_progress_end - 2, f"Decoding chunk {i+1}")
            
            print(f"  Generation complete: output shape={output_ids.shape}")
            
            # Decode
            transcript_ids = output_ids[0, prompt_len:].cpu().tolist()
            text = self.tokenizer.decode(transcript_ids, skip_special_tokens=True).strip()
            print(f"  Decoded text: {text}")
            
            all_transcriptions.append(text)
            
            # Clear memory
            del model_inputs, output_ids, batch
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
        
        if progress_callback:
            progress_callback(90, "Combining results")
        
        final_text = " ".join(all_transcriptions)
        print(f"Transcription complete: {final_text}")
        
        if progress_callback:
            progress_callback(95, "Finalizing")
        
        return {
            "text": final_text,
            "segments": []
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get GLM-ASR capabilities"""
        return {
            "name": "GLM-ASR Nano",
            "supports_timestamps": False,
            "supports_word_timestamps": False,
            "supports_languages": ["auto"],  # GLM-ASR auto-detects
            "max_audio_length": 60 * 60  # 1 hour
        }