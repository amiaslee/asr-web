import mlx_whisper
from pathlib import Path
from typing import Dict, Any, Optional
from base_service import BaseASRService


class MLXWhisperService(BaseASRService):
    """MLX-optimized Whisper service for Apple Silicon"""
    
    def __init__(self, model_name="mlx-community/whisper-large-v3-turbo"):
        self.model_name = model_name
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load Whisper model using MLX (optimized for Apple Silicon)"""
        print(f"Loading MLX Whisper model...")
        try:
            # MLX Whisper loads models on-demand, no explicit loading needed
            # Just verify MLX is available
            import mlx.core as mx
            print(f"✓ MLX Whisper ready (Apple Silicon optimized)")
            self.model_loaded = True
        except Exception as e:
            print(f"✗ Error initializing MLX Whisper: {e}")
            self.model_loaded = False
    
    def is_loaded(self) -> bool:
        """Check if model is ready"""
        return self.model_loaded
    
    def transcribe(self, audio_path: str, language: str = None, word_timestamps: bool = False, progress_callback=None) -> Dict[str, Any]:
        """Transcribe audio file using MLX Whisper with progress tracking
        
        Args:
            audio_path: Path to audio file
            language: Language code or None for auto-detection
            word_timestamps: Whether to include word-level timestamps
            progress_callback: Optional callable(progress: int, status: str)
        """
        if not self.is_loaded():
            raise RuntimeError("MLX Whisper not initialized")
        
        if progress_callback:
            progress_callback(5, "Initializing Whisper model")
        
        print(f"Transcribing with MLX Whisper (word_timestamps={word_timestamps})...")
        
        if progress_callback:
            progress_callback(15, "Loading audio")
        
        try:
            # Detect language if not specified
            if not language:
                if progress_callback:
                    progress_callback(25, "Detecting language")
                
                # Quick detection pass
                detected = mlx_whisper.transcribe(
                    audio_path,
                    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                    language=None,
                    word_timestamps=False,
                    verbose=False
                )
                language = detected.get("language", "en")
                print(f"Detected language: {language}")
                
                if progress_callback:
                    progress_callback(35, f"Language detected: {language}")
            else:
                if progress_callback:
                    progress_callback(35, f"Transcribing in {language}")
            
            # Full transcription with progress
            if progress_callback:
                progress_callback(40, "Running transcription model")
            
            result = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                language=language,
                word_timestamps=word_timestamps,
                verbose=True
            )
            
            if progress_callback:
                progress_callback(75, "Processing segments")
            
            # Debug logging
            if result.get("segments") and len(result["segments"]) > 0:
                print(f"DEBUG: First segment keys: {result['segments'][0].keys()}")
                if "words" in result["segments"][0]:
                    print(f"DEBUG: Has {len(result['segments'][0]['words'])} words")
                    if len(result["segments"][0]["words"]) > 0:
                        print(f"DEBUG: First word: {result['segments'][0]['words'][0]}")
            
            if progress_callback:
                progress_callback(85, "Formatting results")
            
            # Format segments
            segments = []
            if result.get("segments"):
                total_segments = len(result["segments"])
                for idx, seg in enumerate(result["segments"]):
                    if progress_callback and total_segments > 10 and idx % max(1, total_segments // 10) == 0:
                        seg_progress = 85 + (idx * 10) // total_segments
                        progress_callback(seg_progress, f"Formatting segment {idx+1}/{total_segments}")
                    
                    segment_data = {
                        "id": seg.get("id", idx),
                        "start": float(seg["start"]),
                        "end": float(seg["end"]),
                        "text": seg["text"].strip()
                    }
                    
                    # Add word-level timestamps if available
                    if word_timestamps and "words" in seg and seg["words"]:
                        segment_data["words"] = [
                            {
                                "word": w["word"],
                                "start": float(w["start"]),
                                "end": float(w["end"])
                            }
                            for w in seg["words"]
                        ]
                    
                    segments.append(segment_data)
            
            if progress_callback:
                progress_callback(95, "Complete")
            
            return {
                "text": result.get("text", ""),
                "segments": segments,
                "language": result.get("language", language)
            }
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _format_response(self, result: dict, timestamp_level: str) -> Dict[str, Any]:
        """Format MLX Whisper output to unified response format"""
        text = result.get("text", "")
        
        # Build response
        response = {
            "text": text.strip(),
            "segments": [],
            "metadata": {
                "model": "whisper-large-v3-turbo (MLX)",
                "language": result.get("language", "auto")
            }
        }
        
        # Add segments with timestamps if available
        if timestamp_level != "none" and "segments" in result:
            for idx, segment in enumerate(result["segments"]):
                seg = {
                    "id": idx,
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0),
                    "text": segment.get("text", "").strip()
                }
                
                # Add word-level timestamps if available
                if timestamp_level == "word" and "words" in segment:
                    seg["words"] = [
                        {
                            "word": word.get("word", ""),
                            "start": word.get("start", 0.0),
                            "end": word.get("end", 0.0)
                        }
                        for word in segment["words"]
                    ]
                
                response["segments"].append(seg)
        
        return response
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get MLX Whisper capabilities"""
        return {
            "name": "Whisper v3 Turbo (MLX)",
            "supports_timestamps": True,
            "supports_word_timestamps": True,
            "supports_languages": ["auto", "zh", "en", "ja", "ko", "es", "fr", "de", "ru", "ar", "pt"],
            "max_audio_length": 30 * 60,  # 30 minutes
            "optimized_for": "Apple Silicon"
        }
