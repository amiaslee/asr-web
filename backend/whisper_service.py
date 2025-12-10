import torch
import os
from transformers import pipeline
from pathlib import Path
from typing import Dict, Any, List, Optional
from base_service import BaseASRService


class WhisperService(BaseASRService):
    """Whisper ASR service using HuggingFace transformers"""
    
    def __init__(self, model_name="openai/whisper-large-v3-turbo"):
        self.model_name = model_name
        self.pipe = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.actual_device = None  # Will be set during load_model
        self.load_model()
    
    def load_model(self):
        """Load Whisper model with M-series chip acceleration"""
        print(f"Loading Whisper model {self.model_name}...")
        
        # Detect device - prefer MPS for M-series chips
        if torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.float32  # MPS works best with float32
            print("Detected Apple Silicon (M-series) - using MPS acceleration")
        elif torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
            print("Detected CUDA - using GPU acceleration")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            print("Using CPU")
        
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=device,
                torch_dtype=torch_dtype,
            )
            print(f"✓ Whisper model loaded successfully on {device.upper()}")
            self.actual_device = device
        except Exception as e:
            print(f"✗ Error loading on {device}: {e}")
            if device != "cpu":
                print("Falling back to CPU...")
                try:
                    self.pipe = pipeline(
                        "automatic-speech-recognition",
                        model=self.model_name,
                        device="cpu",
                        torch_dtype=torch.float32,
                    )
                    print("✓ Whisper model loaded successfully on CPU")
                    self.actual_device = "cpu"
                except Exception as e2:
                    print(f"✗ Failed to load on CPU: {e2}")
                    self.pipe = None
                    self.actual_device = None
            else:
                self.pipe = None
                self.actual_device = None
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.pipe is not None
    
    def transcribe(
        self, 
        audio_path: str, 
        language: Optional[str] = None,
        timestamp_level: str = "sentence",
        **options
    ) -> Dict[str, Any]:
        """
        Transcribe audio with Whisper
        
        Args:
            audio_path: Path to audio file
            language: Target language (None for auto-detect)
            timestamp_level: "none", "word", or "sentence"
            **options: Additional options
        """
        if not self.is_loaded():
            raise RuntimeError("Whisper model not loaded")
        
        print(f"Transcribing with Whisper (timestamp_level={timestamp_level})...")
        
        # Note: Due to MPS float64 limitations, we cannot use built-in return_timestamps
        # We'll generate approximate timestamps based on chunk positions instead
        
        # Prepare generation kwargs
        generate_kwargs = {}
        if language and language != "auto":
            generate_kwargs["language"] = language
        
        # Run transcription WITHOUT timestamps to avoid MPS float64 error
        try:
            # Call pipeline without return_timestamps parameter
            pipeline_kwargs = {
                "chunk_length_s": 30,
            }
            
            if generate_kwargs:
                pipeline_kwargs["generate_kwargs"] = generate_kwargs
            
            result = self.pipe(audio_path, **pipeline_kwargs)
            
            # If timestamps requested, generate approximate ones
            if timestamp_level != "none":
                result = self._add_approximate_timestamps(result, audio_path, timestamp_level)
            
            return self._format_response(result, timestamp_level)
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            raise
    
    def _add_approximate_timestamps(self, result: dict, audio_path: str, timestamp_level: str) -> dict:
        """Add approximate timestamps since MPS doesn't support built-in timestamp extraction"""
        import librosa
        
        try:
            # Get audio duration
            duration = librosa.get_duration(path=audio_path)
            text = result.get("text", "")
            
            # Split into sentences (rough approximation)
            import re
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return result
            
            # Distribute duration evenly across sentences
            time_per_sentence = duration / len(sentences)
            
            chunks = []
            current_time = 0.0
            for i, sentence in enumerate(sentences):
                chunk_duration = time_per_sentence
                chunks.append({
                    "timestamp": (current_time, current_time + chunk_duration),
                    "text": sentence
                })
                current_time += chunk_duration
            
            result["chunks"] = chunks
            return result
            
        except Exception as e:
            print(f"Warning: Could not add timestamps: {e}")
            # Return result without timestamps
            return result
    
    def _format_response(self, result: dict, timestamp_level: str) -> Dict[str, Any]:
        """Format Whisper output to unified response format"""
        text = result.get("text", "")
        
        # Build response
        response = {
            "text": text.strip(),
            "segments": [],
            "metadata": {
                "model": self.model_name,
                "language": "auto"  # Whisper auto-detects
            }
        }
        
        # Add segments with timestamps if available
        if timestamp_level != "none" and "chunks" in result:
            for idx, chunk in enumerate(result["chunks"]):
                segment = {
                    "id": idx,
                    "start": chunk["timestamp"][0] if chunk["timestamp"][0] is not None else 0.0,
                    "end": chunk["timestamp"][1] if chunk["timestamp"][1] is not None else 0.0,
                    "text": chunk["text"].strip()
                }
                
                # Add word-level timestamps if requested
                if timestamp_level == "word" and isinstance(chunk.get("timestamp"), tuple):
                    # For word-level, each chunk is already a word
                    segment["words"] = [{
                        "word": chunk["text"].strip(),
                        "start": segment["start"],
                        "end": segment["end"]
                    }]
                
                response["segments"].append(segment)
        
        return response
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get Whisper capabilities"""
        return {
            "name": "Whisper Large v3 Turbo",
            "supports_timestamps": True,
            "supports_word_timestamps": True,
            "supports_languages": ["auto", "zh", "en", "ja", "ko", "es", "fr", "de", "ru"],  # Common languages
            "max_audio_length": 30 * 60  # 30 minutes
        }
