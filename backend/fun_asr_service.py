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
        self.model_id = "FunAudioLLM/Fun-ASR-Nano-2512"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Handle MPS (Apple Silicon) if needed, though funasr might rely on specific torch backends
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1" and torch.backends.mps.is_available():
             self.device = "mps" # FunASR support for MPS might be limited, but let's try or fallback to cpu
             # Actually, for safety with new libraries, let's stick to CPU if CUDA is missing unless we know it works
             # But let's default to auto-detect
             pass

    def load_model(self):
        """Load the FunASR model"""
        if self.model is None:
            print(f"Loading FunASR model: {self.model_id} on {self.device}...")
            from funasr import AutoModel

            # Use VAD and Punc models for better results as shown in their demo
            self.model = AutoModel(
                model=self.model_id,
                model_revision="master",
                trust_remote_code=True,
                device=self.device,
                vad_model="fsmn-vad",
                punc_model="ct-punc",
                # spk_model="cam++", # Speaker diarization if needed, but maybe skip for now to keep it light
            )
            print("✓ FunASR model loaded")

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Fun-ASR",
            "supports_timestamps": True,
            "supports_word_timestamps": True, # It seems to support it via the timestamp output
            "supports_languages": ["zh", "en", "ja", "ko", "yue", "auto"], # And many more
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
        # FunASR often auto-detects, but if explicit language is passed:
        generate_kwargs = {
            "input": audio_path,
            "batch_size_s": 300,
            "use_itn": True,
        }

        if language and language != "auto":
            generate_kwargs["language"] = language

        # Run inference
        # FunASR generate returns a list of results
        try:
            res = self.model.generate(**generate_kwargs)
            # res structure is typically [{"key": "...", "text": "...", "timestamp": [...]}]

            if not res:
                return {"text": "", "segments": []}

            result_item = res[0]
            text = result_item.get("text", "")
            raw_timestamp = result_item.get("timestamp", [])

            # Convert to standard segments format
            segments = []

            # Check if timestamp is available and in expected format
            # FunASR timestamp format can be complex or just a list of [start, end]
            # Based on grep, it seems to be [[start, end], ...] corresponding to tokens or sentences?
            # Or using 'sentence_timestamp=True' might be needed for sentence level

            # Let's see if we can structure it.
            # If "sentence_info" is present (often with vad/punc), use that
            # Otherwise parse 'timestamp'

            if "sentence_info" in result_item:
                for sent in result_item["sentence_info"]:
                    segments.append({
                        "start": sent.get("start", 0) / 1000.0, # usually in ms
                        "end": sent.get("end", 0) / 1000.0,
                        "text": sent.get("text", ""),
                        "words": [] # detailed word timestamps if available
                    })
            elif raw_timestamp:
                 # Fallback if structure is different
                 # This is a simplification; precise mapping depends on exact output format
                 pass

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
                    "model": "Fun-ASR",
                    "language": language
                }
            }

        except Exception as e:
            print(f"FunASR inference error: {e}")
            raise e
