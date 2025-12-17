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
    
    def __init__(self, model_service, model_name: str = "fun-asr", buffer_duration: float = 0.5):
        """
        Args:
            model_service: ASR service instance or manager
            model_name: Name of the model to use (defaulting to fun-asr)
            buffer_duration: Seconds of audio to buffer before processing (reduced for lower latency)
                             For streaming (FunASR), this effectively controls the chunk size sent to the model.
        """
        self.model_service = model_service
        self.model_name = model_name
        self.buffer_duration = buffer_duration
        self.sample_rate = 16000
        self.audio_buffer = []
        self.stream_cache = {} # Cache for streaming state (FunASR)
        
    async def handle_connection(self, websocket: WebSocket):
        """Handle WebSocket connection for real-time transcription"""
        await websocket.accept()
        
        # Reset cache on new connection
        self.stream_cache = {}

        try:
            # Send connection confirmation
            await websocket.send_json({
                "type": "connected",
                "message": "Ready for audio streaming (Fun-ASR)",
                "buffer_duration": self.buffer_duration
            })
            
            while True:
                # Receive audio chunk
                data = await websocket.receive()
                
                if "text" in data:
                    # Handle control messages
                    msg = json.loads(data["text"])
                    if msg.get("type") == "stop":
                        # If streaming, we might need to send a final block or clear cache
                        self.stream_cache = {}
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
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            except:
                pass
        finally:
            # Clean up
            self.audio_buffer = []
            self.stream_cache = {}
            
    async def _process_audio_chunk(self, websocket: WebSocket, audio_bytes: bytes):
        """Process incoming audio chunk"""
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Check if model supports true streaming (stream_step)
            service = self.model_service
            if hasattr(self.model_service, "get_model"):
                service = self.model_service.get_model(self.model_name)
                
            if hasattr(service, "stream_step"):
                # Use true streaming for FunASR
                # Pass chunk directly to model
                # We might want to buffer slightly to match model chunk size preference (e.g. 600ms)
                # But let's try just passing what we get if it's reasonable, or use the buffer.

                self.audio_buffer.extend(audio_data)

                # Process if we have enough data (e.g. 200ms or 0.2s)
                # FunASR streaming examples often use chunk_size 200ms+
                required_samples = int(0.2 * self.sample_rate)

                if len(self.audio_buffer) >= required_samples:
                    # Extract chunk
                    chunk_to_process = np.array(self.audio_buffer, dtype=np.float32)
                    self.audio_buffer = [] # Clear buffer completely for streaming

                    # Call stream_step
                    result = await asyncio.to_thread(
                        service.stream_step,
                        audio_chunk=chunk_to_process,
                        cache=self.stream_cache,
                        is_final=False
                    )

                    text = result.get("text", "")
                    if text.strip():
                        await websocket.send_json({
                            "type": "transcription",
                            "text": text,
                            "timestamp": asyncio.get_event_loop().time()
                        })
            else:
                # Legacy buffering logic for models without streaming support
                self.audio_buffer.extend(audio_data)

                # Check if we have enough audio
                buffer_samples = len(self.audio_buffer)
                required_samples = int(self.buffer_duration * self.sample_rate)

                if buffer_samples >= required_samples:
                    # Process buffered audio
                    await self._transcribe_buffer(websocket)

                    # Overlap logic
                    overlap_samples = int(0.5 * self.sample_rate)
                    if len(self.audio_buffer) > overlap_samples:
                        self.audio_buffer = self.audio_buffer[-overlap_samples:]
                    else:
                        self.audio_buffer = []
                
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Audio processing error: {str(e)}"
            })
            
    async def _transcribe_buffer(self, websocket: WebSocket):
        """Transcribe buffered audio (Legacy/File-based)"""
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
            transcribe_kwargs = {"audio_path": tmp_path}
            
            if hasattr(self.model_service, "get_model"):
                 transcribe_kwargs["model_name"] = self.model_name

            result = await asyncio.to_thread(
                self.model_service.transcribe,
                **transcribe_kwargs
            )
            
            # Clean up temp file
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            # Send result
            text = result.get("text", "")
            if text.strip():
                await websocket.send_json({
                    "type": "transcription",
                    "text": text,
                    "timestamp": asyncio.get_event_loop().time()
                })
            
        except Exception as e:
            print(f"Error transcribing buffer: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Transcription error: {str(e)}"
            })
