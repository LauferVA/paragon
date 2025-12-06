import { useEffect, useRef, useCallback } from 'react';
import { useGraphStore } from '../stores/graphStore';
import { getWebSocketUrl } from '../utils/apiClient';
import { GraphWSInbound, GraphWSOutbound } from '../types/websocket.types';

const RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_DELAY_MS = 30000;

export function useGraphWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelayRef = useRef(RECONNECT_DELAY_MS);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const {
    setSnapshot,
    applyDelta,
    setConnected,
    colorMode,
  } = useGraphStore();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const url = getWebSocketUrl('/api/viz/ws');
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[GraphWS] Connected');
      setConnected(true);
      reconnectDelayRef.current = RECONNECT_DELAY_MS;

      // Send initial color mode preference
      const msg: GraphWSOutbound = { type: 'color_mode', mode: colorMode };
      ws.send(JSON.stringify(msg));
    };

    ws.onmessage = (event) => {
      try {
        const message: GraphWSInbound = JSON.parse(event.data);

        switch (message.type) {
          case 'snapshot':
            setSnapshot(message.data);
            break;
          case 'delta':
            applyDelta(message.data);
            break;
          case 'ping':
            ws.send(JSON.stringify({ type: 'pong', data: { timestamp: new Date().toISOString() } }));
            break;
          case 'error':
            console.error('[GraphWS] Error:', message.data.message);
            break;
        }
      } catch (err) {
        console.error('[GraphWS] Parse error:', err);
      }
    };

    ws.onclose = () => {
      console.log('[GraphWS] Disconnected');
      setConnected(false);

      // Exponential backoff reconnection
      reconnectTimeoutRef.current = setTimeout(() => {
        reconnectDelayRef.current = Math.min(
          reconnectDelayRef.current * 2,
          MAX_RECONNECT_DELAY_MS
        );
        connect();
      }, reconnectDelayRef.current);
    };

    ws.onerror = (err) => {
      console.error('[GraphWS] Error:', err);
    };
  }, [setSnapshot, applyDelta, setConnected, colorMode]);

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

  const sendColorMode = useCallback((mode: 'type' | 'status') => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const msg: GraphWSOutbound = { type: 'color_mode', mode };
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return { sendColorMode };
}
