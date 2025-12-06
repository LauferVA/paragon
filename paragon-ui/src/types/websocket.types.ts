/**
 * WebSocket Message Type Definitions for Paragon UI
 * Corresponds to api/websocket.py message schemas
 */

import { GraphSnapshot, GraphDelta, VizNode } from './graph.types';
import { DialecticState, DialecticTurn } from './dialectic.types';

// Inbound messages (from server to client)
export type GraphWSInbound =
  | { type: 'snapshot'; data: GraphSnapshot }
  | { type: 'delta'; data: GraphDelta }
  | { type: 'node_update'; data: VizNode }
  | { type: 'error'; data: { message: string; code?: string } }
  | { type: 'ping'; data: { timestamp: string } };

export type DialecticWSInbound =
  | { type: 'state_update'; data: DialecticState }
  | { type: 'new_turn'; data: DialecticTurn }
  | { type: 'question'; data: { question_id: string; question: string } }
  | { type: 'synthesis'; data: { requirement_nodes: string[]; summary: string } }
  | { type: 'error'; data: { message: string; code?: string } }
  | { type: 'ping'; data: { timestamp: string } };

// Outbound messages (from client to server)
export type GraphWSOutbound =
  | { type: 'subscribe'; data: { filter?: string } }
  | { type: 'unsubscribe'; data: {} }
  | { type: 'request_snapshot'; data: {} }
  | { type: 'pong'; data: { timestamp: string } };

export type DialecticWSOutbound =
  | { type: 'start_session'; data: { requirement_text: string } }
  | { type: 'submit_answer'; data: { question_id: string; answer: string } }
  | { type: 'request_state'; data: {} }
  | { type: 'pause_session'; data: {} }
  | { type: 'resume_session'; data: {} }
  | { type: 'pong'; data: { timestamp: string } };

// WebSocket connection state
export type WSConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error' | 'reconnecting';

export interface WSConnectionInfo {
  state: WSConnectionState;
  url: string;
  reconnect_attempts: number;
  last_error?: string;
  connected_at?: string;
  disconnected_at?: string;
}

// Message envelope for all WebSocket communications
export interface WSMessage<T> {
  type: string;
  data: T;
  timestamp: string;
  sequence?: number;
}
