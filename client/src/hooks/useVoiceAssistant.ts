import { useState, useRef, useCallback, useEffect } from 'react';
import { AudioCaptureManager } from '../audio/capture';
import { AudioQueuePlayer } from '../audio/player';

export type AssistantState =
  | 'IDLE'
  | 'LISTENING'
  | 'TRANSCRIBING'
  | 'THINKING'
  | 'SPEAKING'
  | 'INTERRUPTED';

export interface TelemetryMetrics {
  stt_ms: number;
  ttft_ms: number;
  tts_first_chunk_ms: number;
  total_e2e_ms: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  language?: string;
  timestamp: Date;
}

export function useVoiceAssistant() {
  const [state, setState] = useState<AssistantState>('IDLE');
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [telemetry, setTelemetry] = useState<TelemetryMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [micVolume, setMicVolume] = useState<number>(0);

  const wsRef = useRef<WebSocket | null>(null);
  const captureRef = useRef<AudioCaptureManager | null>(null);
  const playerRef = useRef<AudioQueuePlayer | null>(null);

  const getWsUrl = () => {
    if (import.meta.env.VITE_WS_URL) {
      return import.meta.env.VITE_WS_URL;
    }
    const host = window.location.hostname || 'localhost';
    return `ws://${host}:8000/ws`;
  };

  const handlePlaybackEnded = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'playback_ended' }));
    }
  }, []);

  const stopCall = useCallback(() => {
    if (captureRef.current) {
      captureRef.current.stop();
      captureRef.current = null;
    }

    if (playerRef.current) {
      playerRef.current.close();
      playerRef.current = null;
    }

    if (wsRef.current) {
      try {
        wsRef.current.send(JSON.stringify({ type: 'stop' }));
        wsRef.current.close();
      } catch {
        // Ignored
      }
      wsRef.current = null;
    }

    setIsConnected(false);
    setMicVolume(0);
    setState('IDLE');
  }, []);

  const startCall = useCallback(async () => {
    setError(null);
    try {
      // 1. Initialize Audio Player
      const player = new AudioQueuePlayer(handlePlaybackEnded);
      playerRef.current = player;

      // 2. Connect WebSocket
      const wsUrl = getWsUrl();
      const ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onmessage = (event) => {
        // Binary frame = 24kHz audio from Kokoro TTS
        if (event.data instanceof ArrayBuffer) {
          player.queueAudioChunk(event.data);
          return;
        }

        // Text frame = control / telemetry JSON
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'state_change') {
            setState(msg.state as AssistantState);
          } else if (msg.type === 'flush') {
            player.flush();
            setState('INTERRUPTED');
          } else if (msg.type === 'transcript') {
            setMessages((prev) => [
              ...prev,
              {
                id: Math.random().toString(36).substring(7),
                role: msg.role,
                text: msg.text,
                language: msg.language,
                timestamp: new Date(),
              },
            ]);
          } else if (msg.type === 'telemetry') {
            setTelemetry({
              stt_ms: msg.stt_ms,
              ttft_ms: msg.ttft_ms,
              tts_first_chunk_ms: msg.tts_first_chunk_ms,
              total_e2e_ms: msg.total_e2e_ms,
            });
          }
        } catch (err) {
          console.error('Error parsing JSON message:', err);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        setMicVolume(0);
        setState('IDLE');
      };

      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => {
          setIsConnected(true);
          resolve();
        };
        ws.onerror = (_e) => {
          reject(new Error(`WebSocket connection failed to ${wsUrl}. Is the backend running?`));
        };
      });

      // 3. Initialize continuous microphone capture
      const capture = new AudioCaptureManager();
      captureRef.current = capture;

      await capture.start(
        (chunk: ArrayBuffer) => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(chunk);
          }
        },
        (volume: number) => {
          setMicVolume(volume);
        }
      );

      setState('LISTENING');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      stopCall();
    }
  }, [handlePlaybackEnded, stopCall]);

  useEffect(() => {
    return () => {
      stopCall();
    };
  }, [stopCall]);

  return {
    state,
    isConnected,
    messages,
    telemetry,
    micVolume,
    error,
    startCall,
    stopCall,
  };
}
