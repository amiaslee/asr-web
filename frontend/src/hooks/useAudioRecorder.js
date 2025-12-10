import { useState, useRef, useCallback } from 'react';

/**
 * Hook for recording audio from microphone using Web Audio API
 * Extracts raw PCM samples for WebSocket streaming
 */
export const useAudioRecorder = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [duration, setDuration] = useState(0);
    const [error, setError] = useState(null);

    const audioContextRef = useRef(null);
    const processorRef = useRef(null);
    const streamRef = useRef(null);
    const durationIntervalRef = useRef(null);
    const onDataAvailableRef = useRef(null);

    const startRecording = useCallback(async (onDataAvailable) => {
        try {
            setError(null);

            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            streamRef.current = stream;
            onDataAvailableRef.current = onDataAvailable;

            // Create AudioContext
            const audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });
            audioContextRef.current = audioContext;

            // Create audio source from microphone stream
            const source = audioContext.createMediaStreamSource(stream);

            // Create ScriptProcessorNode for PCM extraction
            // Buffer size: 4096 samples = ~256ms at 16kHz
            const processor = audioContext.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;

            processor.onaudioprocess = (event) => {
                if (!onDataAvailableRef.current) return;

                // Get PCM samples (Float32Array, values between -1 and 1)
                const inputBuffer = event.inputBuffer;
                const channelData = inputBuffer.getChannelData(0);

                // Convert Float32 to Int16 (backend expects int16)
                const int16Data = new Int16Array(channelData.length);
                for (let i = 0; i < channelData.length; i++) {
                    // Clamp values to [-1, 1] and scale to int16 range
                    const sample = Math.max(-1, Math.min(1, channelData[i]));
                    int16Data[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
                }

                // Send as Blob for compatibility with existing WebSocket code
                const blob = new Blob([int16Data.buffer], { type: 'application/octet-stream' });
                onDataAvailableRef.current(blob);
            };

            // Connect nodes: source -> processor -> destination
            source.connect(processor);
            processor.connect(audioContext.destination);

            setIsRecording(true);

            // Start duration counter
            setDuration(0);
            durationIntervalRef.current = setInterval(() => {
                setDuration(d => d + 1);
            }, 1000);

        } catch (err) {
            console.error('Error starting recording:', err);
            setError(err.message);
            throw err;
        }
    }, []);

    const stopRecording = useCallback(() => {
        if (isRecording) {
            // Disconnect and clean up processor
            if (processorRef.current) {
                processorRef.current.disconnect();
                processorRef.current = null;
            }

            // Close audio context
            if (audioContextRef.current) {
                audioContextRef.current.close();
                audioContextRef.current = null;
            }

            // Stop all audio tracks
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
                streamRef.current = null;
            }

            // Clear duration interval
            if (durationIntervalRef.current) {
                clearInterval(durationIntervalRef.current);
                durationIntervalRef.current = null;
            }

            onDataAvailableRef.current = null;
            setIsRecording(false);
        }
    }, [isRecording]);

    const pauseRecording = useCallback(() => {
        if (audioContextRef.current && isRecording) {
            audioContextRef.current.suspend();
            if (durationIntervalRef.current) {
                clearInterval(durationIntervalRef.current);
            }
        }
    }, [isRecording]);

    const resumeRecording = useCallback(() => {
        if (audioContextRef.current && isRecording) {
            audioContextRef.current.resume();
            durationIntervalRef.current = setInterval(() => {
                setDuration(d => d + 1);
            }, 1000);
        }
    }, [isRecording]);

    return {
        isRecording,
        duration,
        error,
        startRecording,
        stopRecording,
        pauseRecording,
        resumeRecording
    };
};
