import { useState, useRef, useCallback, useEffect } from 'react';

/**
 * Hook for WebSocket connection to real-time transcription service
 */
export const useWebSocketTranscription = () => {
    const [isConnected, setIsConnected] = useState(false);
    const [transcriptions, setTranscriptions] = useState([]);
    const [error, setError] = useState(null);

    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);

    const connect = useCallback((model = 'glm-asr') => {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            // Bypass proxy and connect directly to backend (port 8000)
            // Note: This requires CORS to be enabled on backend (which it is via CORSMiddleware)
            const hostname = window.location.hostname;
            const wsUrl = `${protocol}//${hostname}:8000/ws/transcribe?model=${model}`;
            console.log(`Attempting Direct WebSocket connection to: ${wsUrl}`);

            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('WebSocket connected');
                setIsConnected(true);
                setError(null);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.type === 'connected') {
                        console.log('Connection confirmed:', data.message);
                    } else if (data.type === 'transcription') {
                        setTranscriptions(prev => [...prev, {
                            text: data.text,
                            timestamp: data.timestamp,
                            id: Date.now()
                        }]);
                    } else if (data.type === 'error') {
                        console.error('Server error:', data.message);
                        setError(data.message);
                    }
                } catch (err) {
                    console.error('Error parsing message:', err);
                }
            };

            ws.onerror = (event) => {
                console.error('WebSocket error:', event);
                setError('Connection error');
            };

            ws.onclose = () => {
                console.log('WebSocket closed');
                setIsConnected(false);
                wsRef.current = null;
            };

        } catch (err) {
            console.error('Error connecting:', err);
            setError(err.message);
        }
    }, []);

    const disconnect = useCallback(() => {
        if (wsRef.current) {
            // Send stop message
            if (wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({ type: 'stop' }));
            }
            wsRef.current.close();
            wsRef.current = null;
        }
        setIsConnected(false);
    }, []);

    const sendAudio = useCallback(async (audioBlob) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            try {
                // Convert audio blob to ArrayBuffer
                const arrayBuffer = await audioBlob.arrayBuffer();
                wsRef.current.send(arrayBuffer);
            } catch (err) {
                console.error('Error sending audio:', err);
                setError('Failed to send audio');
            }
        }
    }, []);

    const clearTranscriptions = useCallback(() => {
        setTranscriptions([]);
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            disconnect();
        };
    }, [disconnect]);

    return {
        isConnected,
        transcriptions,
        error,
        connect,
        disconnect,
        sendAudio,
        clearTranscriptions
    };
};
