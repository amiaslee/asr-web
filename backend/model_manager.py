from typing import Dict, Optional
from base_service import BaseASRService


class ModelManager:
    """Manages multiple ASR models and provides unified access"""
    
    def __init__(self):
        self.models: Dict[str, BaseASRService] = {}
        self.default_model = "glm-asr"
    
    def register_model(self, name: str, service: BaseASRService):
        """Register a new ASR model"""
        self.models[name] = service
        print(f"Registered model: {name}")
    
    def get_model(self, model_name: Optional[str] = None) -> Optional[BaseASRService]:
        """Get a specific model or the default model"""
        name = model_name or self.default_model
        return self.models.get(name)
    
    def list_models(self) -> list:
        """List all available models with their capabilities"""
        result = []
        for name, service in self.models.items():
            # Always get capabilities (they don't require the model to be loaded)
            capabilities = service.get_capabilities()
            result.append({
                "key": name,  # Registration key (for API calls)
                "name": capabilities.get("name", name),  # Display name
                "loaded": service.is_loaded(),  # Actual loaded status
                "supports_timestamps": capabilities.get("supports_timestamps", False),
                "supports_word_timestamps": capabilities.get("supports_word_timestamps", False),
                "supports_languages": capabilities.get("supports_languages", []),
                "max_audio_length": capabilities.get("max_audio_length", 0)
            })
        return result
    
    def get_model_names(self) -> list:
        """Get list of all registered model names"""
        return list(self.models.keys())
    
    def all_loaded(self) -> bool:
        """Check if all models are loaded"""
        return all(service.is_loaded() for service in self.models.values())


# Global model manager instance
model_manager = ModelManager()
