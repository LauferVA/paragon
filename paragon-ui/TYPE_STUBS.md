# Type Stubs for Paragon UI

**Purpose:** Quick reference for implementing type definitions required by components
**Status:** REFERENCE - Extract from FRONTEND_SPEC.md Section 3

---

## Required Type Files

### 1. /src/types/graph.types.ts

```typescript
// Matches VizNode from backend (viz/core.py)
export interface VizNode {
  id: string;
  type: string;
  status: string;
  label: string;
  color: string;
  size: number;
  x?: number;
  y?: number;
  created_by: string;
  created_at: string;
  teleology_status: string;
  layer: number;
  is_root: boolean;
  is_leaf: boolean;
}

// Matches VizEdge from backend
export interface VizEdge {
  source: string;
  target: string;
  type: string;
  color: string;
  weight: number;
}

// Matches GraphSnapshot from backend
export interface GraphSnapshot {
  timestamp: string;
  node_count: number;
  edge_count: number;
  nodes: VizNode[];
  edges: VizEdge[];
  layer_count: number;
  has_cycle: boolean;
  root_count: number;
  leaf_count: number;
  version: string;
  label: string;
}

// Matches GraphDelta from backend
export interface GraphDelta {
  timestamp: string;
  sequence: number;
  nodes_added: VizNode[];
  nodes_updated: VizNode[];
  nodes_removed: string[];
  edges_added: VizEdge[];
  edges_removed: [string, string][];
}
```

---

### 2. /src/types/dialectic.types.ts

```typescript
// Matches AmbiguityMarker from backend (agents/schemas.py)
export interface AmbiguityMarker {
  category: 'SUBJECTIVE' | 'COMPARATIVE' | 'UNDEFINED_PRONOUN' | 'UNDEFINED_TERM' | 'MISSING_CONTEXT';
  text: string;
  impact: 'BLOCKING' | 'CLARIFYING';
  suggested_question: string | null;
  suggested_answer: string | null;
}

// Matches ClarificationQuestion from backend
export interface ClarificationQuestion {
  question: string;
  context: string;
  options: string[];
  priority: 'low' | 'medium' | 'high';
}

// User answer submission payload
export interface ClarificationResponse {
  answers: Record<number, string>;
  session_id: string;
}
```

---

## Required Store Files

### 1. /src/stores/graphStore.ts

**Minimal Implementation:**

```typescript
import { create } from 'zustand';
import { GraphSnapshot, GraphDelta, VizNode, VizEdge } from '../types/graph.types';

interface GraphStore {
  nodes: Map<string, VizNode>;
  edges: Map<string, VizEdge>;
  snapshot: GraphSnapshot | null;
  selectedNodeId: string | null;
  hoveredNodeId: string | null;
  colorMode: 'type' | 'status';
  lastSequence: number;
  isConnected: boolean;

  setSnapshot: (snapshot: GraphSnapshot) => void;
  applyDelta: (delta: GraphDelta) => void;
  setSelectedNode: (nodeId: string | null) => void;
  setHoveredNode: (nodeId: string | null) => void;
  setColorMode: (mode: 'type' | 'status') => void;
  setConnected: (connected: boolean) => void;
  reset: () => void;
}

export const useGraphStore = create<GraphStore>((set, get) => ({
  nodes: new Map(),
  edges: new Map(),
  snapshot: null,
  selectedNodeId: null,
  hoveredNodeId: null,
  colorMode: 'type',
  lastSequence: 0,
  isConnected: false,

  setSnapshot: (snapshot) => {
    const nodes = new Map(snapshot.nodes.map(n => [n.id, n]));
    const edges = new Map(snapshot.edges.map(e => [`${e.source}-${e.target}`, e]));
    set({ snapshot, nodes, edges, lastSequence: 0 });
  },

  applyDelta: (delta) => {
    const { nodes, edges, lastSequence } = get();
    if (delta.sequence <= lastSequence) return;

    const newNodes = new Map(nodes);
    const newEdges = new Map(edges);

    delta.nodes_added.forEach(n => newNodes.set(n.id, n));
    delta.nodes_updated.forEach(n => newNodes.set(n.id, n));
    delta.nodes_removed.forEach(id => newNodes.delete(id));

    delta.edges_added.forEach(e => newEdges.set(`${e.source}-${e.target}`, e));
    delta.edges_removed.forEach(([s, t]) => newEdges.delete(`${s}-${t}`));

    set({ nodes: newNodes, edges: newEdges, lastSequence: delta.sequence });
  },

  setSelectedNode: (nodeId) => set({ selectedNodeId: nodeId }),
  setHoveredNode: (nodeId) => set({ hoveredNodeId: nodeId }),
  setColorMode: (mode) => set({ colorMode: mode }),
  setConnected: (connected) => set({ isConnected: connected }),
  reset: () => set({
    nodes: new Map(),
    edges: new Map(),
    snapshot: null,
    selectedNodeId: null,
    hoveredNodeId: null,
    lastSequence: 0,
    isConnected: false,
  }),
}));
```

---

### 2. /src/stores/dialecticStore.ts

**Minimal Implementation:**

```typescript
import { create } from 'zustand';
import { AmbiguityMarker, ClarificationQuestion } from '../types/dialectic.types';

interface DialecticStore {
  phase: 'DIALECTIC' | 'CLARIFICATION' | 'RESEARCH' | 'IDLE';
  ambiguities: AmbiguityMarker[];
  questions: ClarificationQuestion[];
  answers: Map<number, string>;
  pendingSubmit: boolean;
  isConnected: boolean;

  setPhase: (phase: DialecticStore['phase']) => void;
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
```

---

### 3. /src/stores/uiStore.ts

**Minimal Implementation:**

```typescript
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UIStore {
  theme: 'dark' | 'light';
  splitRatio: number;
  metricsCollapsed: boolean;
  legendCollapsed: boolean;

  setTheme: (theme: 'dark' | 'light') => void;
  setSplitRatio: (ratio: number) => void;
  toggleMetrics: () => void;
  toggleLegend: () => void;
}

export const useUIStore = create<UIStore>()(
  persist(
    (set) => ({
      theme: 'dark',
      splitRatio: 0.6,
      metricsCollapsed: false,
      legendCollapsed: false,

      setTheme: (theme) => set({ theme }),
      setSplitRatio: (ratio) => set({ splitRatio: ratio }),
      toggleMetrics: () => set((state) => ({ metricsCollapsed: !state.metricsCollapsed })),
      toggleLegend: () => set((state) => ({ legendCollapsed: !state.legendCollapsed })),
    }),
    {
      name: 'paragon-ui-settings',
    }
  )
);
```

---

## Mock Data for Testing

### Mock GraphSnapshot

```typescript
import { GraphSnapshot } from './types/graph.types';

export const mockSnapshot: GraphSnapshot = {
  timestamp: new Date().toISOString(),
  node_count: 3,
  edge_count: 2,
  nodes: [
    {
      id: 'REQ-001',
      type: 'REQ',
      status: 'VERIFIED',
      label: 'Fast sorting function',
      color: '#ef4444',
      size: 10,
      created_by: 'user',
      created_at: new Date().toISOString(),
      teleology_status: 'COMPLETE',
      layer: 0,
      is_root: true,
      is_leaf: false,
    },
    {
      id: 'SPEC-001',
      type: 'SPEC',
      status: 'VERIFIED',
      label: 'QuickSort specification',
      color: '#f97316',
      size: 10,
      created_by: 'architect_agent',
      created_at: new Date().toISOString(),
      teleology_status: 'COMPLETE',
      layer: 1,
      is_root: false,
      is_leaf: false,
    },
    {
      id: 'CODE-001',
      type: 'CODE',
      status: 'PENDING',
      label: 'quicksort.py',
      color: '#14b8a6',
      size: 10,
      created_by: 'builder_agent',
      created_at: new Date().toISOString(),
      teleology_status: 'PARTIAL',
      layer: 2,
      is_root: false,
      is_leaf: true,
    },
  ],
  edges: [
    {
      source: 'REQ-001',
      target: 'SPEC-001',
      type: 'IMPLEMENTS',
      color: '#14b8a6',
      weight: 1.0,
    },
    {
      source: 'SPEC-001',
      target: 'CODE-001',
      type: 'IMPLEMENTS',
      color: '#14b8a6',
      weight: 1.0,
    },
  ],
  layer_count: 3,
  has_cycle: false,
  root_count: 1,
  leaf_count: 1,
  version: '1.0.0',
  label: 'test-snapshot',
};
```

### Mock Ambiguities

```typescript
import { AmbiguityMarker } from './types/dialectic.types';

export const mockAmbiguities: AmbiguityMarker[] = [
  {
    category: 'SUBJECTIVE',
    text: 'fast sorting function',
    impact: 'CLARIFYING',
    suggested_question: 'What performance target do you have in mind?',
    suggested_answer: 'O(n log n) time complexity, handling 1M elements in under 1 second',
  },
  {
    category: 'UNDEFINED_TERM',
    text: 'enterprise-grade',
    impact: 'BLOCKING',
    suggested_question: 'What specific features define "enterprise-grade"?',
    suggested_answer: null,
  },
];
```

---

## Quick Start

1. Copy type definitions to `/src/types/` files
2. Copy store implementations to `/src/stores/` files
3. Test components with mock data
4. Connect to backend API
5. Integrate WebSocket for real-time updates

---

**Reference:** See FRONTEND_SPEC.md sections 3-4 for complete specifications
**Components Ready:** All 8 components in `/src/components/`
