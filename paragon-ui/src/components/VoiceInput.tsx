import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useUIStore } from '../stores/uiStore';

interface VoiceInputProps {
  onTranscript?: (text: string) => void;
  onError?: (error: string) => void;
  className?: string;
  disabled?: boolean;
}

interface TranscriptMessage {
  type: 'ready' | 'transcript' | 'complete' | 'error' | 'ping';
  message?: string;
  model?: string;
  data?: {
    text: string;
    start: number;
    end: number;
    confidence: number;
    is_final: boolean;
    language: string;
  };
}

/**
 * VoiceInput Component
 *
 * Provides a microphone button for real-time voice-to-text transcription.
 *
 * Features:
 * - Real-time audio streaming via WebSocket
 * - Visual recording indicator
 * - Automatic transcript insertion
 * - Error handling and fallback
 *
 * Usage:
 *   <VoiceInput onTranscript={(text) => console.log(text)} />
 */
export function VoiceInput({
  onTranscript,
  onError,
  className = '',
  disabled = false,
}: VoiceInputProps) {
  const { theme } = useUIStore();

  const [isRecording, setIsRecording] = useState(false);
  const [isAvailable, setIsAvailable] = useState(false);
  const [status, setStatus] = useState<string>('Checking availability...');
  const [transcriptBuffer, setTranscriptBuffer] = useState<string>('');

  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  // Check if STT service is available on mount
  useEffect(() => {
    checkAvailability();
  }, []);

  const checkAvailability = async () => {
    try {
      const response = await fetch('/api/stt/status');
      const data = await response.json();

      if (data.available) {
        setIsAvailable(true);
        setStatus(`Ready (${data.model})`);
      } else {
        setIsAvailable(false);
        setStatus('Service unavailable');
        onError?.('Speech-to-text service is not available');
      }
    } catch (error) {
      setIsAvailable(false);
      setStatus('Connection error');
      onError?.('Failed to connect to speech-to-text service');
    }
  };

  const startRecording = useCallback(async () => {
    if (!isAvailable || isRecording) return;

    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1, // Mono
          sampleRate: 16000, // 16kHz recommended for Whisper
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      streamRef.current = stream;

      // Create WebSocket connection
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/audio`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('Connected, preparing audio...');
      };

      ws.onmessage = (event) => {
        const msg: TranscriptMessage = JSON.parse(event.data);

        if (msg.type === 'ready') {
          setStatus(`Recording (${msg.model})...`);
          setIsRecording(true);
          setTranscriptBuffer('');
        } else if (msg.type === 'transcript') {
          // Accumulate transcript
          const text = msg.data?.text || '';
          setTranscriptBuffer((prev) => (prev ? prev + ' ' + text : text));
        } else if (msg.type === 'complete') {
          setStatus('Transcription complete');
          if (transcriptBuffer && onTranscript) {
            onTranscript(transcriptBuffer);
          }
        } else if (msg.type === 'error') {
          setStatus(`Error: ${msg.message}`);
          onError?.(msg.message || 'Unknown error');
          stopRecording();
        }
      };

      ws.onerror = (error) => {
        setStatus('WebSocket error');
        onError?.('Connection error during recording');
        stopRecording();
      };

      ws.onclose = () => {
        setStatus('Connection closed');
        if (isRecording) {
          stopRecording();
        }
      };

      // Create AudioContext for resampling
      audioContextRef.current = new AudioContext({ sampleRate: 16000 });

      // Create MediaRecorder
      // Note: We need to convert browser audio to the format expected by the server
      // Browser typically gives us 48kHz, we need 16kHz 16-bit PCM

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus', // Best browser support
      });

      mediaRecorderRef.current = mediaRecorder;

      // Buffer audio chunks and send via WebSocket
      mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
          // Convert the audio blob to the format expected by the server
          const arrayBuffer = await event.data.arrayBuffer();

          // For simplicity, we'll send the raw audio data
          // The server will handle format conversion
          // In production, you might want to resample here
          ws.send(arrayBuffer);
        }
      };

      mediaRecorder.onerror = (error) => {
        console.error('MediaRecorder error:', error);
        onError?.('Recording error');
        stopRecording();
      };

      // Start recording - send chunks every second
      mediaRecorder.start(1000);

    } catch (error) {
      console.error('Failed to start recording:', error);
      setStatus('Microphone access denied');
      onError?.(
        error instanceof Error
          ? error.message
          : 'Failed to access microphone'
      );
    }
  }, [isAvailable, isRecording, onTranscript, onError, transcriptBuffer]);

  const stopRecording = useCallback(() => {
    // Stop media recorder
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }

    // Stop all tracks in the stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // Send stop signal and close WebSocket
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'stop' }));
      wsRef.current.close();
    }

    wsRef.current = null;
    mediaRecorderRef.current = null;
    setIsRecording(false);

    // Send final transcript
    if (transcriptBuffer && onTranscript) {
      onTranscript(transcriptBuffer);
    }

    setStatus(isAvailable ? 'Ready' : 'Service unavailable');
  }, [isAvailable, transcriptBuffer, onTranscript]);

  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (isRecording) {
        stopRecording();
      }
    };
  }, [isRecording, stopRecording]);

  // Theme-based colors
  const buttonBaseClass = theme === 'dark' ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-200 hover:bg-gray-300';
  const buttonActiveClass = theme === 'dark' ? 'bg-red-700 hover:bg-red-600' : 'bg-red-500 hover:bg-red-600';
  const buttonDisabledClass = theme === 'dark' ? 'bg-gray-800 text-gray-600' : 'bg-gray-100 text-gray-400';
  const textClass = theme === 'dark' ? 'text-gray-300' : 'text-gray-700';
  const tooltipClass = theme === 'dark' ? 'bg-gray-800 text-gray-200 border-gray-700' : 'bg-white text-gray-800 border-gray-300';

  return (
    <div className={`inline-flex items-center gap-2 ${className}`}>
      <button
        onClick={toggleRecording}
        disabled={disabled || !isAvailable}
        className={`
          relative p-3 rounded-full transition-all duration-200
          ${disabled || !isAvailable ? buttonDisabledClass : isRecording ? buttonActiveClass : buttonBaseClass}
          ${isRecording ? 'animate-pulse' : ''}
          focus:outline-none focus:ring-2 focus:ring-blue-500
        `}
        title={isRecording ? 'Stop recording' : 'Start voice input'}
        aria-label={isRecording ? 'Stop recording' : 'Start voice input'}
      >
        {/* Microphone Icon */}
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className={`w-5 h-5 ${disabled || !isAvailable ? '' : 'text-white'}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
          />
        </svg>

        {/* Recording indicator dot */}
        {isRecording && (
          <span className="absolute top-0 right-0 w-3 h-3 bg-red-500 rounded-full border-2 border-white" />
        )}
      </button>

      {/* Status text (optional) */}
      {status && (
        <span className={`text-xs ${textClass}`}>
          {status}
        </span>
      )}

      {/* Transcript preview (optional) */}
      {transcriptBuffer && isRecording && (
        <div className={`absolute top-full mt-2 p-2 rounded shadow-lg border max-w-xs ${tooltipClass}`}>
          <div className="text-xs font-medium mb-1">Transcript:</div>
          <div className="text-sm">{transcriptBuffer}</div>
        </div>
      )}
    </div>
  );
}
