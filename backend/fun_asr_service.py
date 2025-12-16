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
                spk_model="cam++",
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

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio using FunASR
        """
        if not self.model:
            self.load_model()

        # Prepare arguments
        language = kwargs.get("language", "auto")
        generate_kwargs = {
            "input": audio_path,
            "batch_size_s": 300,
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
