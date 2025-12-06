import { useEffect, useRef, useCallback } from 'react';
import { useDialecticStore } from '../stores/dialecticStore';
import { getWebSocketUrl } from '../utils/apiClient';
import { DialecticWSInbound, DialecticWSOutbound } from '../types/websocket.types';
import { ClarificationAnswer } from '../types/dialectic.types';

const RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_DELAY_MS = 30000;

export function useDialecticWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelayRef = useRef(RECONNECT_DELAY_MS);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const {
    setPhase,
    setAmbiguities,
    setQuestions,
    setPendingSubmit,
    setConnected,
    answers,
  } = useDialecticStore();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const url = getWebSocketUrl('/api/dialectic/ws');
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[DialecticWS] Connected');
      setConnected(true);
      reconnectDelayRef.current = RECONNECT_DELAY_MS;
    };

    ws.onmessage = (event) => {
      try {
        const message: DialecticWSInbound = JSON.parse(event.data);

        switch (message.type) {
          case 'state_update':
            setPhase(message.data.current_phase as any);
            setAmbiguities(message.data.ambiguities);
            setQuestions(message.data.questions);
            break;
          case 'new_turn':
            // Handle new turn in dialectic conversation
            break;
          case 'question':
            // Handle individual question
            break;
          case 'synthesis':
            setPhase('RESEARCH');
            break;
          case 'ping':
            ws.send(JSON.stringify({ type: 'pong', data: { timestamp: new Date().toISOString() } }));
            break;
          case 'error':
            console.error('[DialecticWS] Error:', message.data.message);
            break;
        }
      } catch (err) {
        console.error('[DialecticWS] Parse error:', err);
      }
    };

    ws.onclose = () => {
      console.log('[DialecticWS] Disconnected');
      setConnected(false);

      reconnectTimeoutRef.current = setTimeout(() => {
        reconnectDelayRef.current = Math.min(
          reconnectDelayRef.current * 2,
          MAX_RECONNECT_DELAY_MS
        );
        connect();
      }, reconnectDelayRef.current);
    };

    ws.onerror = (err) => {
      console.error('[DialecticWS] Error:', err);
    };
  }, [setPhase, setAmbiguities, setQuestions, setConnected]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const submitAnswers = useCallback((sessionId: string) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      console.error('[DialecticWS] Not connected');
      return;
    }

    setPendingSubmit(true);

    const answersObj: Record<string, ClarificationAnswer> = {};
    answers.forEach((answer, index) => {
      answersObj[index.toString()] = {
        question_id: index.toString(),
        answer,
        confidence: 1.0,
        timestamp: new Date().toISOString(),
      };
    });

    const msg: DialecticWSOutbound = {
      type: 'submit_answer',
      data: { question_id: sessionId, answer: JSON.stringify(answersObj) },
    };
    wsRef.current.send(JSON.stringify(msg));
  }, [answers, setPendingSubmit]);

  const startSession = useCallback((requirementText: string) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      console.error('[DialecticWS] Not connected');
      return;
    }

    const msg: DialecticWSOutbound = {
      type: 'start_session',
      data: { requirement_text: requirementText },
    };
    wsRef.current.send(JSON.stringify(msg));
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return { submitAnswers, startSession };
}
