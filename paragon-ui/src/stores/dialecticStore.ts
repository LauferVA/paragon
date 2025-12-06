import { create } from 'zustand';
import { AmbiguityMarker, ClarificationQuestion } from '../types/dialectic.types';

type DialecticPhase = 'DIALECTIC' | 'CLARIFICATION' | 'RESEARCH' | 'IDLE';

interface DialecticStore {
  // State
  phase: DialecticPhase;
  ambiguities: AmbiguityMarker[];
  questions: ClarificationQuestion[];
  answers: Map<number, string>;
  pendingSubmit: boolean;
  isConnected: boolean;

  // Actions
  setPhase: (phase: DialecticPhase) => void;
  setAmbiguities: (ambiguities: AmbiguityMarker[]) => void;
  setQuestions: (questions: ClarificationQuestion[]) => void;
  setAnswer: (index: number, answer: string) => void;
  acceptSuggested: (index: number, suggested: string) => void;
  clearAnswer: (index: number) => void;
  setPendingSubmit: (pending: boolean) => void;
  setConnected: (connected: boolean) => void;
  reset: () => void;
}

export const useDialecticStore = create<DialecticStore>((set, get) => ({
  phase: 'IDLE',
  ambiguities: [],
  questions: [],
  answers: new Map(),
  pendingSubmit: false,
  isConnected: false,

  setPhase: (phase) => set({ phase }),
  setAmbiguities: (ambiguities) => set({ ambiguities }),
  setQuestions: (questions) => set({ questions }),
  setAnswer: (index, answer) => {
    const { answers } = get();
    const newAnswers = new Map(answers);
    newAnswers.set(index, answer);
    set({ answers: newAnswers });
  },
  acceptSuggested: (index, suggested) => {
    const { answers } = get();
    const newAnswers = new Map(answers);
    newAnswers.set(index, suggested);
    set({ answers: newAnswers });
  },
  clearAnswer: (index) => {
    const { answers } = get();
    const newAnswers = new Map(answers);
    newAnswers.delete(index);
    set({ answers: newAnswers });
  },
  setPendingSubmit: (pending) => set({ pendingSubmit: pending }),
  setConnected: (connected) => set({ isConnected: connected }),
  reset: () => set({
    phase: 'IDLE',
    ambiguities: [],
    questions: [],
    answers: new Map(),
    pendingSubmit: false,
    isConnected: false,
  }),
}));
