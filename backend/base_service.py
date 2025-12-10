from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseASRService(ABC):
    """Abstract base class for all ASR services"""
    
    @abstractmethod
    def load_model(self):
        """Load and initialize the ASR model"""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready"""
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: str, **options) -> Dict[str, Any]:
        """
        Transcribe audio file to text with optional features
        
        Args:
            audio_path: Path to audio file
            **options: Additional options (language, timestamp_level, etc.)
            
        Returns:
            Dictionary with transcription result:
            {
                "text": str,
                "segments": List[dict],
                "metadata": dict
            }
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get model capabilities
        
        Returns:
            Dictionary with capabilities:
            {
                "name": str,
                "supports_timestamps": bool,
                "supports_word_timestamps": bool,
                "supports_languages": List[str],
                "max_audio_length": int (seconds)
            }
        """
        pass
