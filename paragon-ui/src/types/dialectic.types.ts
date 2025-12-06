/**
 * Dialectic Type Definitions for Paragon UI
 * Corresponds to agents/schemas.py dialectic models
 */

export interface AmbiguityMarker {
  id: string;
  text: string;
  type: 'semantic' | 'scope' | 'technical' | 'requirement';
  severity: 'low' | 'medium' | 'high' | 'critical';
  position?: {
    line?: number;
    column?: number;
    context?: string;
  };
  suggested_clarifications?: string[];
}

export interface ClarificationQuestion {
  id: string;
  question: string;
  context: string;
  options?: string[];
  requires_freeform: boolean;
  related_ambiguity_ids: string[];
  priority: number;
}

export interface ClarificationAnswer {
  question_id: string;
  answer: string;
  confidence: number;
  timestamp: string;
}

export interface DialecticState {
  session_id: string;
  status: 'active' | 'paused' | 'completed' | 'abandoned';
  current_phase: 'analysis' | 'questioning' | 'clarification' | 'synthesis';
  ambiguities: AmbiguityMarker[];
  questions: ClarificationQuestion[];
  answers: ClarificationAnswer[];
  resolved_count: number;
  total_count: number;
  started_at: string;
  updated_at: string;
}

export interface DialecticTurn {
  turn_number: number;
  agent: 'system' | 'user';
  type: 'question' | 'answer' | 'observation' | 'synthesis';
  content: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface DialecticSession {
  id: string;
  requirement_text: string;
  state: DialecticState;
  turns: DialecticTurn[];
  created_at: string;
  updated_at: string;
}

export interface DialecticMetrics {
  total_ambiguities: number;
  resolved_ambiguities: number;
  critical_ambiguities: number;
  avg_resolution_time_ms: number;
  question_count: number;
  answer_count: number;
}
