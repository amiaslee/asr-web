from typing import Dict, Any, Optional
import os
import torch
from base_service import BaseASRService

class FunASRService(BaseASRService):
    """
    ASR Service using FunAudioLLM/Fun-ASR (via funasr package)
    """

    def __init__(self):
        self.model = None
        # Use Paraformer Streaming as it is the most stable and widely used streaming model for FunASR
        # Fun-ASR-Nano-2512 seems to have deployment issues with missing Qwen3 files on ModelScope currently.
        # Paraformer is excellent for real-time applications.
        self.model_id = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Handle MPS (Apple Silicon) if needed, though funasr might rely on specific torch backends
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1" and torch.backends.mps.is_available():
             self.device = "mps"

    def load_model(self):
        """Load the FunASR model"""
        if self.model is None:
            print(f"Loading FunASR model: {self.model_id} on {self.device}...")
            from funasr import AutoModel

            # Use VAD and Punc models for better results
            self.model = AutoModel(
                model=self.model_id,
                model_revision="master",
                trust_remote_code=True,
                device=self.device,
                vad_model="fsmn-vad",
                punc_model="ct-punc",
                # spk_model="cam++", # Streaming model does not support timestamp prediction needed for diarization
            )
            print("✓ FunASR model loaded")

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Fun-ASR (Paraformer)",
            "supports_timestamps": True,
            "supports_word_timestamps": True,
            "supports_languages": ["zh", "en", "auto"], # Paraformer is strong in ZH/EN
            "description": "FunAudioLLM End-to-End Speech Recognition"
        }

    def stream_step(self, audio_chunk: Any, cache: Dict[str, Any] = None, is_final: bool = False) -> Dict[str, Any]:
        """
        Streaming step for FunASR
        """
        if not self.model:
            self.load_model()

        # Input audio chunk should be a numpy array or bytes
        # FunASR streaming expects input as numpy array

        try:
            # We assume audio_chunk is already correct format (numpy array usually)
            # The model is iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online

            # The streaming API:
            # res = model.generate(input=chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, ...)

            # Default chunk size for paraformer online
            # According to docs: [0, 10, 5] means 600ms latency configuration?
            # Or simplified: just pass the chunk.

            # Note: For online models, cache is updated in-place or returned?
            # In FunASR `AutoModel.generate`, if input is not file, it returns result.
            # And it updates cache if passed?
            # Looking at docs: res = model.generate(..., cache=cache, ...)

            res = self.model.generate(
                input=audio_chunk,
                cache=cache if cache is not None else {},
                is_final=is_final,
                # chunk_size=[0, 10, 5], # Let's use default or omit if possible
                # encoder_chunk_look_back=4,
                # decoder_chunk_look_back=1
            )

            # res structure:
            # It usually returns a list of results.
            # For streaming, it returns results for the current chunk.

            text = ""
            if res:
                # Depending on the model output format.
                # Usually: [{'text': '...', 'timestamp': ...}]
                result_item = res[0]
                text = result_item.get("text", "")

            return {
                "text": text,
                # "segments": ... # Timestamp info might be partial
            }

        except Exception as e:
            print(f"FunASR streaming error: {e}")
            # If error, return empty
            return {"text": ""}

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio using FunASR (File-based)
        """
        if not self.model:
            self.load_model()

        # Prepare arguments
        language = kwargs.get("language", "auto")
        generate_kwargs = {
            "input": audio_path,
            "batch_size": 1, # Streaming model requires batch_size=1
            "use_itn": True,
        }

        # Run inference
        try:
            res = self.model.generate(**generate_kwargs)

            if not res:
                return {"text": "", "segments": []}

            result_item = res[0]
            text = result_item.get("text", "")

            # Convert to standard segments format
            segments = []

            if "sentence_info" in result_item:
                for sent in result_item["sentence_info"]:
                    segments.append({
                        "start": sent.get("start", 0) / 1000.0, # usually in ms
                        "end": sent.get("end", 0) / 1000.0,
                        "text": sent.get("text", ""),
                        "words": [] # detailed word timestamps if available
                    })

            # If segments were empty, try to construct from raw text
            if not segments and text:
                segments.append({
                    "start": 0,
                    "end": 0, # Unknown
                    "text": text
                })

            return {
                "text": text,
                "segments": segments,
                "metadata": {
                    "model": "Fun-ASR (Paraformer)",
                    "language": language
                }
            }

        except Exception as e:
            print(f"FunASR inference error: {e}")
            raise e
