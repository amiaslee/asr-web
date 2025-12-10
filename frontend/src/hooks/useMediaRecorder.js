import { useState, useRef, useCallback } from 'react';

/**
 * Hook for recording audio using MediaRecorder
 * Generates WebM files suitable for playback and download
 */
export const useMediaRecorder = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [duration, setDuration] = useState(0);
    const [error, setError] = useState(null);

    const mediaRecorderRef = useRef(null);
    const streamRef = useRef(null);
    const chunksRef = useRef([]);
    const durationIntervalRef = useRef(null);

    const startRecording = useCallback(async () => {
        try {
            setError(null);
            chunksRef.current = [];

            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                }
            });

            streamRef.current = stream;

            // Create MediaRecorder with appropriate format
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : 'audio/webm';

            const mediaRecorder = new MediaRecorder(stream, {
                mimeType,
                audioBitsPerSecond: 128000
            });

            mediaRecorderRef.current = mediaRecorder;

            // Collect data chunks
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    chunksRef.current.push(event.data);
                }
            };

            // Start recording - request data every second
            mediaRecorder.start(1000);
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
        return new Promise((resolve) => {
            if (mediaRecorderRef.current && isRecording) {
                mediaRecorderRef.current.onstop = () => {
                    // Combine all chunks
                    const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
                    resolve(blob);

                    // Clean up
                    chunksRef.current = [];
                };

                mediaRecorderRef.current.stop();
                setIsRecording(false);

                // Stop all tracks
                if (streamRef.current) {
                    streamRef.current.getTracks().forEach(track => track.stop());
                    streamRef.current = null;
                }

                // Clear duration interval
                if (durationIntervalRef.current) {
                    clearInterval(durationIntervalRef.current);
                    durationIntervalRef.current = null;
                }

                mediaRecorderRef.current = null;
            } else {
                resolve(null);
            }
        });
    }, [isRecording]);

    return {
        isRecording,
        duration,
        error,
        startRecording,
        stopRecording
    };
};
