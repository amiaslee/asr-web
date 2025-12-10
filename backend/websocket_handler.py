"""
WebSocket handler for real-time audio transcription
"""
import asyncio
import json
import base64
import io
import soundfile as sf
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from typing import Optional


class RealtimeTranscriptionHandler:
    """Handle real-time audio transcription via WebSocket"""
    
    def __init__(self, model_service, model_name: str = "glm-asr", buffer_duration: float = 3.0):
        """
        Args:
            model_service: ASR service instance or manager
            model_name: Name of the model to use
            buffer_duration: Seconds of audio to buffer before processing
        """
        self.model_service = model_service
        self.model_name = model_name
        self.buffer_duration = buffer_duration
        self.sample_rate = 16000
        self.audio_buffer = []
        
    async def handle_connection(self, websocket: WebSocket):
        """Handle WebSocket connection for real-time transcription"""
        await websocket.accept()
        
        try:
            # Send connection confirmation
            await websocket.send_json({
                "type": "connected",
                "message": "Ready for audio streaming",
                "buffer_duration": self.buffer_duration
            })
            
            while True:
                # Receive audio chunk
                data = await websocket.receive()
                
                if "text" in data:
                    # Handle control messages
                    msg = json.loads(data["text"])
                    if msg.get("type") == "stop":
                        break
                    elif msg.get("type") == "config":
                        # Future: handle configuration updates
                        pass
                        
                elif "bytes" in data:
                    # Process audio data
                    await self._process_audio_chunk(websocket, data["bytes"])
                    
        except WebSocketDisconnect:
            print("Client disconnected")
        except Exception as e:
            print(f"Error in WebSocket handler: {e}")
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        finally:
            # Clean up
            self.audio_buffer = []
            
    async def _process_audio_chunk(self, websocket: WebSocket, audio_bytes: bytes):
        """Process incoming audio chunk"""
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add to buffer
            self.audio_buffer.extend(audio_data)
            
            # Check if we have enough audio
            buffer_samples = len(self.audio_buffer)
            required_samples = int(self.buffer_duration * self.sample_rate)
            
            if buffer_samples >= required_samples:
                # Process buffered audio
                await self._transcribe_buffer(websocket)
                
                # Keep last 0.5s for context
                overlap_samples = int(0.5 * self.sample_rate)
                self.audio_buffer = self.audio_buffer[-overlap_samples:]
                
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Audio processing error: {str(e)}"
            })
            
    async def _transcribe_buffer(self, websocket: WebSocket):
        """Transcribe buffered audio"""
        try:
            # Convert buffer to audio file in memory
            audio_array = np.array(self.audio_buffer, dtype=np.float32)
            
            # Save to temporary bytes buffer
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, self.sample_rate, format='WAV')
            buffer.seek(0)
            
            # Save to temp file for model processing
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(buffer.read())
                tmp_path = tmp_file.name
            
            # Transcribe
            # Transcribe
            # Only pass timestamp_level if model supports it
            capabilities = self.model_service.get_capabilities(self.model_name)
            transcribe_kwargs = {"audio_path": tmp_path}
            
            # If model_service is a manager (has get_model), we might need to pass model_name to transcribe
            # or better, get the model instance first if possible. 
            # But assuming model_service.transcribe handles dispatch.
            if hasattr(self.model_service, "get_model"):
                 transcribe_kwargs["model_name"] = self.model_name

            if capabilities.get("supports_timestamps", False):
                transcribe_kwargs["timestamp_level"] = "none"  # Faster without timestamps for real-time
            
            result = await asyncio.to_thread(
                self.model_service.transcribe,
                **transcribe_kwargs
            )
            
            # Clean up temp file
            import os
            os.unlink(tmp_path)
            
            # Send result
            await websocket.send_json({
                "type": "transcription",
                "text": result.get("text", ""),
                "timestamp": asyncio.get_event_loop().time()
            })
            
        except Exception as e:
            print(f"Error transcribing buffer: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Transcription error: {str(e)}"
            })
