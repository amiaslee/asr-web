#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# CRITICAL: Set MPS fallback BEFORE importing any torch/transformers modules
# This fixes float64 errors on Apple Silicon M-series chips
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
import shutil
import asyncio
import json

# Import services
from glm_asr_service import GLMASRService
from mlx_whisper_service import MLXWhisperService

app = FastAPI(title="GLM-ASR Web API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelManager:
    """Manages multiple ASR models with lazy loading"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {
            "glm-asr": {
                "class": GLMASRService,
                "loaded": False,
                "capabilities": {
                    "language_selection": False,
                    "timestamp_level": False,
                    "max_tokens": True
                }
            },
            "whisper-turbo": {
                "class": MLXWhisperService,
                "loaded": False,
                "capabilities": {
                    "language_selection": True,
                    "timestamp_level": True,
                    "max_tokens": False
                }
            }
        }
        self.default_model = "glm-asr"
        print("Model manager initialized (models will load on first use)")
    
    def get_model(self, model_name: str):
        """Get model with lazy loading - only load when first requested"""
        if model_name not in self.model_configs:
            return None
        
        # Load model if not already loaded
        if model_name not in self.models:
            print(f"Loading model {model_name}...")
            config = self.model_configs[model_name]
            self.models[model_name] = config["class"]()
            config["loaded"] = True
            print(f"✓ {model_name} loaded successfully")
        
        return self.models[model_name]
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if model is loaded"""
        return self.model_configs.get(model_name, {}).get("loaded", False)
    
    def get_capabilities(self, model_name: str) -> dict:
        """Get capabilities for a specific model"""
        if model_name not in self.model_configs:
            return {}
        return self.model_configs[model_name]["capabilities"]
    
    def get_model_names(self):
        """Get list of available model names"""
        return list(self.model_configs.keys())

    def all_loaded(self) -> bool:
        """Check if all registered models are loaded"""
        return all(config["loaded"] for config in self.model_configs.values())

    def transcribe(self, audio_path: str, model_name: str = None, **kwargs):
        """Proxy transcribe call to the appropriate model service"""
        # If model_name is not provided, use default
        if not model_name:
            model_name = self.default_model
            
        # Get the model instance (loads it if not loaded)
        service = self.get_model(model_name)
        if not service:
            raise ValueError(f"Model {model_name} not found")
            
        # Remove model_name from kwargs if it made it there to avoid unexpected argument errors in service
        if "model_name" in kwargs:
            kwargs.pop("model_name")
            
        # Delegate to the model service
        return service.transcribe(audio_path, **kwargs)

    def list_models(self):
        """List all available models with their capabilities and loaded status"""
        model_list = []
        for name, config in self.model_configs.items():
            model_list.append({
                "name": name,
                "loaded": config["loaded"],
                "capabilities": config["capabilities"]
            })
        return model_list

# Initialize model manager (models load on first use)
model_manager = ModelManager()

print("Available models:", model_manager.get_model_names())


@app.get("/health")
async def health():
    """Health check endpoint"""
    models_status = {}
    for name in model_manager.get_model_names():
        models_status[name] = model_manager.is_model_loaded(name)
    
    return {
        "status": "ok",
        "models": models_status,
        "all_loaded": all(models_status.values())
    }


@app.get("/models")
async def list_models():
    """List all available models with their capabilities"""
    return {
        "models": model_manager.list_models(),
        "default": model_manager.default_model
    }


async def progress_generator(model_service, file_path: str, **kwargs):
    """Generate Server-Sent Events for transcription progress with real model feedback"""
    try:
        # Send initial progress
        yield f"data: {json.dumps({'progress': 0, 'status': 'Starting transcription'})}\n\n"
        await asyncio.sleep(0.1)
        
        # Progress tracking variables
        current_progress = [0]
        current_status = ["Initializing"]
        
        def progress_callback(progress: int, status: str):
            """Callback function for model services to report progress"""
            current_progress[0] = progress
            current_status[0] = status
        
        # Start transcription in background with progress callback
        kwargs['progress_callback'] = progress_callback
        
        # Run transcription in thread to avoid blocking
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(model_service.transcribe, file_path, **kwargs)
        
        # Poll for progress updates
        last_progress = 0
        while not future.done():
            await asyncio.sleep(0.3)  # Check every 300ms
            
            if current_progress[0] > last_progress:
                yield f"data: {json.dumps({'progress': current_progress[0], 'status': current_status[0]})}\n\n"
                last_progress = current_progress[0]
        
        # Get final result
        result = future.result()
        
        # Send final result
        yield f"data: {json.dumps({'progress': 100, 'status': 'Complete', 'result': result})}\n\n"
        
    except Exception as e:
        print(f"Error in progress_generator: {e}")
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/transcribe-stream")
async def transcribe_stream(
    file: UploadFile = File(...),
    model: str = Query("glm-asr"),
    language: Optional[str] = Query(None),
    timestamp_level: str = Query("none"),
    max_tokens: int = Query(2048)
):
    """Transcribe with real-time progress via Server-Sent Events"""
    
    model_service = model_manager.get_model(model)
    if not model_service:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not available")
    
    # Save uploaded file temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name
    
    try:
        # Prepare kwargs
        kwargs = {}
        if model == "whisper-turbo":
            if language and language != "auto":
                kwargs["language"] = language
            if timestamp_level != "none":
                kwargs["word_timestamps"] = (timestamp_level == "word")
        elif model == "glm-asr":
            kwargs["max_tokens"] = max_tokens
        
        return StreamingResponse(
            progress_generator(model_service, tmp_path, **kwargs),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )
    finally:
        # Cleanup will happen after streaming completes
        pass


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Query("glm-asr"),
    language: Optional[str] = Query(None),
    timestamp_level: str = Query("none"),
    max_tokens: int = Query(2048)
):
    """Transcribe audio/video file - standard endpoint without progress"""
    
    model_service = model_manager.get_model(model)
    if not model_service:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not available")
    
    # Save uploaded file temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name
    
    try:
        # Call the appropriate transcription method
        if model == "whisper-turbo":
            kwargs = {}
            if language and language != "auto":
                kwargs["language"] = language
            if timestamp_level != "none":
                kwargs["word_timestamps"] = (timestamp_level == "word")
            
            result = model_service.transcribe(tmp_path, **kwargs)
        elif model == "glm-asr":
            result = model_service.transcribe(tmp_path, max_tokens=max_tokens)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
        
        return result
    
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass


# WebSocket endpoint for real-time transcription
from websocket_handler import RealtimeTranscriptionHandler

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket, model: str = "glm-asr"):
    """WebSocket endpoint for real-time streaming transcription"""
    handler = RealtimeTranscriptionHandler(model_manager, model_name=model)
    await handler.handle_connection(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
