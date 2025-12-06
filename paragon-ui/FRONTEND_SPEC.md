# PARAGON FRONTEND SPECIFICATION v1.0

**Project:** Paragon UI - Real-Time Graph Visualization & Dialectic Interface
**Technology Stack:** React 18 + TypeScript + Cosmograph + Zustand + TailwindCSS
**Backend:** FastAPI WebSocket API (Starlette)
**Date:** 2025-12-06
**Status:** DETAILED SPECIFICATION - Ready for Implementation

---

## TABLE OF CONTENTS

1. [Architecture Overview](#1-architecture-overview)
2. [Component Specifications](#2-component-specifications)
3. [Props Interfaces (TypeScript)](#3-props-interfaces-typescript)
4. [State Management](#4-state-management)
5. [API Integration](#5-api-integration)
6. [WebSocket Messages](#6-websocket-messages)
7. [Event Handlers](#7-event-handlers)
8. [Implementation Order](#8-implementation-order)

---

## 1. ARCHITECTURE OVERVIEW

### 1.1 Application Structure

```
paragon-ui/
├── public/
│   └── index.html
├── src/
│   ├── App.tsx                      # Root component with layout
│   ├── main.tsx                     # Entry point
│   │
│   ├── components/
│   │   ├── GraphViewer.tsx          # Main Cosmograph integration
│   │   ├── DialecticChat.tsx        # Question/answer interface
│   │   ├── NodeTooltip.tsx          # Hover popup component
│   │   ├── MetricsDashboard.tsx     # Graph statistics panel
│   │   ├── Legend.tsx               # Node/edge type legend
│   │   ├── AmbiguityCard.tsx        # Single ambiguity display
│   │   ├── QuestionCard.tsx         # Clarification question card
│   │   └── Layout.tsx               # Split-pane layout wrapper
│   │
│   ├── hooks/
│   │   ├── useGraphWebSocket.ts     # Graph updates WebSocket
│   │   ├── useDialecticWebSocket.ts # Dialectic phase WebSocket
│   │   ├── useGraphSnapshot.ts      # Initial snapshot fetch
│   │   └── useNodeSelection.ts      # Node click/hover state
│   │
│   ├── stores/
│   │   ├── graphStore.ts            # Zustand store for graph state
│   │   ├── dialecticStore.ts        # Zustand store for chat state
│   │   └── uiStore.ts               # UI preferences (theme, layout)
│   │
│   ├── types/
│   │   ├── graph.types.ts           # Graph domain types
│   │   ├── dialectic.types.ts       # Dialectic domain types
│   │   └── websocket.types.ts       # WebSocket message types
│   │
│   ├── utils/
│   │   ├── arrowParser.ts           # Apache Arrow IPC deserializer
│   │   ├── colorMaps.ts             # Node/edge color constants
│   │   └── graphLayout.ts           # Layout computation helpers
│   │
│   └── styles/
│       └── globals.css              # TailwindCSS imports + custom styles
│
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

### 1.2 Technology Rationale

| Technology | Purpose | Reason |
|------------|---------|--------|
| **React 18** | Component framework | Concurrent rendering for smooth updates |
| **TypeScript** | Type safety | Prevent runtime errors, better DX |
| **Cosmograph** | Graph rendering | GPU-accelerated, handles 10K+ nodes |
| **Zustand** | State management | Lightweight, no boilerplate, good TypeScript support |
| **TailwindCSS** | Styling | Rapid development, dark mode support |
| **Vite** | Build tool | Fast HMR, native ESM |
| **Apache Arrow JS** | Data deserialization | Zero-copy parsing for large graphs |

### 1.3 Design Principles

1. **Performance First:** 60 FPS with 10K+ nodes
2. **Type Safety:** All data flows typed end-to-end
3. **Real-Time:** <100ms latency for graph updates
4. **Accessibility:** WCAG AA compliant, keyboard navigation
5. **Dark Mode Native:** Designed for dark theme (light optional)

---

## 2. COMPONENT SPECIFICATIONS

### 2.1 GraphViewer Component

**File:** `src/components/GraphViewer.tsx`

**Responsibility:**
Primary graph visualization component. Integrates Cosmograph, handles node/edge rendering, manages layout, and dispatches interaction events.

**Children:**
- `NodeTooltip` (positioned absolutely over hovered node)
- `Legend` (fixed position overlay)

**Key Features:**
- GPU-accelerated force-directed layout via Cosmograph
- Real-time delta updates without full re-render
- Configurable color modes (type-based, status-based)
- Click-to-select nodes, hover for tooltips
- Zoom/pan controls
- Hierarchical layout hints (using `layer` property from backend)

**Rendering Strategy:**
- Initial render: Full snapshot from `/api/viz/snapshot` or WebSocket
- Incremental updates: Apply deltas from WebSocket
- Re-layout: Only when graph structure changes significantly

**Performance Targets:**
- Initial render 10K nodes: <5 seconds
- Delta update latency: <100ms
- Frame rate: 60 FPS during interactions

---

### 2.2 DialecticChat Component

**File:** `src/components/DialecticChat.tsx`

**Responsibility:**
Interactive chat interface for the dialectic/clarification phase. Displays ambiguities, presents questions, and collects user answers.

**Children:**
- `AmbiguityCard` (rendered for each detected ambiguity)
- `QuestionCard` (rendered for each clarification question)

**Key Features:**
- Phase indicator (DIALECTIC, CLARIFICATION, RESEARCH)
- Ambiguity list with category badges (SUBJECTIVE, COMPARATIVE, etc.)
- Accept suggested answer or provide custom answer
- Submit all answers as batch
- Scroll to latest message
- Typing indicators during LLM processing

**States:**
1. **DIALECTIC Phase:** Showing detected ambiguities
2. **CLARIFICATION Phase:** Collecting user answers
3. **RESEARCH Phase:** Readonly, showing what was clarified
4. **Idle:** No active session

**User Flows:**
- User sees ambiguity with suggested answer
- Click "Accept Suggested" → auto-fill answer
- Click "Provide Own Answer" → show text input
- Submit all → send to backend via WebSocket

---

### 2.3 NodeTooltip Component

**File:** `src/components/NodeTooltip.tsx`

**Responsibility:**
Displays detailed node information on hover. Positioned near cursor.

**Layout:**
```
┌─────────────────────────────────────┐
│ CODE: calculate_hash                │  ← Title (16px, bold)
│ (Function in crypto/hash.py)        │  ← Subtitle (12px, gray)
├─────────────────────────────────────┤
│ Status: VERIFIED ✓                  │  ← Status badge
│ Created: 2025-12-06 14:32           │  ← Timestamp
│ Agent: builder_agent_1              │  ← Agent ID
├─────────────────────────────────────┤
│ Traces to: REQ-8e6243b6             │  ← Links
│ Implements: SPEC-a1b2c3d4           │
├─────────────────────────────────────┤
│ Click for details                   │  ← Action hint
└─────────────────────────────────────┘
```

**Key Features:**
- Appears on hover with 200ms delay
- Disappears on mouse leave
- Shows node type, status, created_by, created_at
- Shows teleology links (traces_to, implements)
- Keyboard accessible (focus on node with Tab)

---

### 2.4 MetricsDashboard Component

**File:** `src/components/MetricsDashboard.tsx`

**Responsibility:**
Displays live graph statistics in a compact panel.

**Metrics Displayed:**
- Node count (total)
- Edge count (total)
- Layer count (topological depth)
- Root count (nodes with no predecessors)
- Leaf count (nodes with no successors)
- Has cycle (boolean, should always be false)
- Node type breakdown (pie chart or bar chart)
- Status breakdown (pie chart or bar chart)

**Layout:**
- Fixed position: top-right corner
- Collapsible (click to minimize)
- Updates in real-time via Zustand store

---

### 2.5 Legend Component

**File:** `src/components/Legend.tsx`

**Responsibility:**
Displays color legend for node/edge types.

**Layout:**
```
┌──────────────────────┐
│ NODE TYPES           │
│ ● REQ   Requirement  │  ← Red
│ ● SPEC  Specification│  ← Orange
│ ● CODE  Implementation│ ← Teal
│ ● TEST  Test         │  ← Dark blue
│ ● DOC   Documentation│  ← Medium blue
│                      │
│ EDGE TYPES           │
│ ─ IMPLEMENTS         │  ← Teal
│ ─ TRACES_TO          │  ← Red
│ ─ DEPENDS_ON         │  ← Gray
└──────────────────────┘
```

**Key Features:**
- Fixed position: bottom-left corner
- Collapsible
- Synced with color mode (type vs status)

---

### 2.6 AmbiguityCard Component

**File:** `src/components/AmbiguityCard.tsx`

**Responsibility:**
Displays a single detected ambiguity with context.

**Props:**
- `ambiguity: AmbiguityMarker` (from backend schema)
- `onAcceptSuggested: () => void`
- `onProvidOwn: () => void`

**Layout:**
```
┌────────────────────────────────────────────────┐
│ 🟡 SUBJECTIVE: "fast sorting function"        │
│                                                │
│ Question: What performance target?             │
│                                                │
│ Suggested: O(n log n), 1M elements in <1s     │
│                                                │
│ [ Accept Suggested ]  [ Provide Own Answer ]  │
└────────────────────────────────────────────────┘
```

**Key Features:**
- Category badge with icon (🟡 for SUBJECTIVE, 🔴 for BLOCKING)
- Shows ambiguous text highlighted
- Shows clarification question
- Shows suggested answer (if available)
- Two-button interaction

---

### 2.7 QuestionCard Component

**File:** `src/components/QuestionCard.tsx`

**Responsibility:**
Interactive card for collecting user answer to a clarification question.

**States:**
- `suggested_mode`: Showing suggested answer with Accept button
- `custom_mode`: Showing text input for custom answer
- `answered`: Answer submitted (readonly)

**Key Features:**
- Smooth transition between modes
- Text input with validation
- Submit only when answer provided
- Visual feedback on submission

---

### 2.8 Layout Component

**File:** `src/components/Layout.tsx`

**Responsibility:**
Split-pane layout wrapper for GraphViewer and DialecticChat.

**Layout:**
```
┌────────────────────────────────────────────────┐
│  Paragon v1.0              [Dark] [Settings]   │ ← Header
├─────────────────────┬──────────────────────────┤
│                     │                          │
│   GraphViewer       │   DialecticChat          │
│   (60% width)       │   (40% width)            │
│                     │                          │
│                     │                          │
│                     │                          │
│   [Legend]          │   [Phase: DIALECTIC]     │
│   [Metrics]         │   [Input...]             │
│                     │                          │
└─────────────────────┴──────────────────────────┘
```

**Key Features:**
- Resizable divider (drag to adjust width)
- Responsive: stacks vertically on mobile
- Persist layout preferences in localStorage
- Header with app name, version, theme toggle

---

## 3. PROPS INTERFACES (TypeScript)

### 3.1 Graph Domain Types

**File:** `src/types/graph.types.ts`

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

// Client-side graph state (in Zustand store)
export interface GraphState {
  nodes: Map<string, VizNode>;
  edges: Map<string, VizEdge>;
  snapshot: GraphSnapshot | null;
  selectedNodeId: string | null;
  hoveredNodeId: string | null;
  colorMode: 'type' | 'status';
  lastSequence: number;
}
```

### 3.2 Dialectic Domain Types

**File:** `src/types/dialectic.types.ts`

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

// Client-side dialectic state
export interface DialecticState {
  phase: 'DIALECTIC' | 'CLARIFICATION' | 'RESEARCH' | 'IDLE';
  ambiguities: AmbiguityMarker[];
  questions: ClarificationQuestion[];
  answers: Map<number, string>; // question index -> user answer
  pendingSubmit: boolean;
}

// User answer submission payload
export interface ClarificationResponse {
  answers: Record<number, string>;
  session_id: string;
}
```

### 3.3 WebSocket Message Types

**File:** `src/types/websocket.types.ts`

```typescript
// Inbound message types (server -> client)
export type GraphWSInbound =
  | { type: 'snapshot'; data: GraphSnapshot }
  | { type: 'delta'; data: GraphDelta }
  | { type: 'heartbeat' }
  | { type: 'pong' }
  | { type: 'error'; message: string };

// Outbound message types (client -> server)
export type GraphWSOutbound =
  | { type: 'color_mode'; mode: 'type' | 'status' }
  | { type: 'ping' };

// Dialectic WebSocket messages
export type DialecticWSInbound =
  | { type: 'phase_change'; phase: string }
  | { type: 'ambiguities'; data: AmbiguityMarker[] }
  | { type: 'questions'; data: ClarificationQuestion[] }
  | { type: 'complete' }
  | { type: 'error'; message: string };

export type DialecticWSOutbound =
  | { type: 'submit_answers'; data: ClarificationResponse };
```

### 3.4 Component Props

**File:** `src/types/component.types.ts`

```typescript
// GraphViewer
export interface GraphViewerProps {
  width: number;
  height: number;
  colorMode?: 'type' | 'status';
  onNodeClick?: (nodeId: string) => void;
  onNodeHover?: (nodeId: string | null) => void;
}

// DialecticChat
export interface DialecticChatProps {
  sessionId: string;
  onComplete?: () => void;
}

// NodeTooltip
export interface NodeTooltipProps {
  node: VizNode | null;
  position: { x: number; y: number };
  visible: boolean;
}

// MetricsDashboard
export interface MetricsDashboardProps {
  snapshot: GraphSnapshot | null;
  collapsed?: boolean;
  onToggle?: () => void;
}

// Legend
export interface LegendProps {
  colorMode: 'type' | 'status';
  collapsed?: boolean;
  onToggle?: () => void;
}

// AmbiguityCard
export interface AmbiguityCardProps {
  ambiguity: AmbiguityMarker;
  index: number;
  onAcceptSuggested: (index: number, answer: string) => void;
  onProvidOwn: (index: number) => void;
  answer?: string;
}

// QuestionCard
export interface QuestionCardProps {
  question: ClarificationQuestion;
  index: number;
  answer?: string;
  onAnswerChange: (index: number, answer: string) => void;
  disabled?: boolean;
}
```

---

## 4. STATE MANAGEMENT

### 4.1 Graph Store (Zustand)

**File:** `src/stores/graphStore.ts`

```typescript
import { create } from 'zustand';
import { GraphSnapshot, GraphDelta, VizNode, VizEdge } from '../types/graph.types';

interface GraphStore {
  // State
  nodes: Map<string, VizNode>;
  edges: Map<string, VizEdge>;
  snapshot: GraphSnapshot | null;
  selectedNodeId: string | null;
  hoveredNodeId: string | null;
  colorMode: 'type' | 'status';
  lastSequence: number;
  isConnected: boolean;

  // Actions
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

    // Sequence check for ordering
    if (delta.sequence <= lastSequence) {
      console.warn('Out-of-order delta ignored', delta.sequence, lastSequence);
      return;
    }

    const newNodes = new Map(nodes);
    const newEdges = new Map(edges);

    // Apply node changes
    delta.nodes_added.forEach(n => newNodes.set(n.id, n));
    delta.nodes_updated.forEach(n => newNodes.set(n.id, n));
    delta.nodes_removed.forEach(id => newNodes.delete(id));

    // Apply edge changes
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

### 4.2 Dialectic Store (Zustand)

**File:** `src/stores/dialecticStore.ts`

```typescript
import { create } from 'zustand';
import { AmbiguityMarker, ClarificationQuestion } from '../types/dialectic.types';

interface DialecticStore {
  // State
  phase: 'DIALECTIC' | 'CLARIFICATION' | 'RESEARCH' | 'IDLE';
  ambiguities: AmbiguityMarker[];
  questions: ClarificationQuestion[];
  answers: Map<number, string>;
  pendingSubmit: boolean;
  isConnected: boolean;

  // Actions
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

### 4.3 UI Store (Zustand)

**File:** `src/stores/uiStore.ts`

```typescript
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UIStore {
  // State
  theme: 'dark' | 'light';
  splitRatio: number; // 0-1, left panel width ratio
  metricsCollapsed: boolean;
  legendCollapsed: boolean;

  // Actions
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

## 5. API INTEGRATION

### 5.1 REST Endpoints

**Base URL:** `http://localhost:8000` (configurable via env)

| Endpoint | Method | Purpose | Response Type |
|----------|--------|---------|---------------|
| `/health` | GET | Health check | `{ status: string }` |
| `/stats` | GET | Graph statistics | `{ node_count, edge_count, has_cycle }` |
| `/api/viz/snapshot` | GET | Initial graph snapshot | `GraphSnapshot` |
| `/api/viz/stream` | GET | Arrow IPC format snapshot | Binary (Arrow IPC) |
| `/api/viz/snapshots` | GET | List saved snapshots | `{ snapshots: [] }` |

**Query Parameters for `/api/viz/snapshot`:**
- `color_mode`: `"type"` or `"status"` (default: `"type"`)
- `version`: Version label for caching (default: `"current"`)

### 5.2 HTTP Client Configuration

**File:** `src/utils/apiClient.ts`

```typescript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function fetchSnapshot(colorMode: 'type' | 'status' = 'type'): Promise<GraphSnapshot> {
  const response = await fetch(`${API_BASE_URL}/api/viz/snapshot?color_mode=${colorMode}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch snapshot: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchStats(): Promise<{ node_count: number; edge_count: number; has_cycle: boolean }> {
  const response = await fetch(`${API_BASE_URL}/stats`);
  if (!response.ok) {
    throw new Error(`Failed to fetch stats: ${response.statusText}`);
  }
  return response.json();
}
```

---

## 6. WEBSOCKET MESSAGES

### 6.1 Graph WebSocket (`/api/viz/ws`)

**Connection:** `ws://localhost:8000/api/viz/ws`

#### Server → Client Messages

```typescript
// 1. Initial snapshot on connection
{
  type: 'snapshot',
  data: GraphSnapshot
}

// 2. Incremental update
{
  type: 'delta',
  data: GraphDelta
}

// 3. Heartbeat (every 30s if no activity)
{
  type: 'heartbeat'
}

// 4. Pong (response to client ping)
{
  type: 'pong'
}

// 5. Error
{
  type: 'error',
  message: string
}
```

#### Client → Server Messages

```typescript
// 1. Request color mode change
{
  type: 'color_mode',
  mode: 'type' | 'status'
}

// 2. Ping (for keepalive)
{
  type: 'ping'
}
```

### 6.2 Dialectic WebSocket (Future: `/api/dialectic/ws`)

**Note:** This endpoint is NOT yet implemented in the backend. The specification is prepared for future implementation.

#### Server → Client Messages

```typescript
// 1. Phase change notification
{
  type: 'phase_change',
  phase: 'DIALECTIC' | 'CLARIFICATION' | 'RESEARCH'
}

// 2. Ambiguities detected
{
  type: 'ambiguities',
  data: AmbiguityMarker[]
}

// 3. Clarification questions
{
  type: 'questions',
  data: ClarificationQuestion[]
}

// 4. Dialectic phase complete
{
  type: 'complete'
}

// 5. Error
{
  type: 'error',
  message: string
}
```

#### Client → Server Messages

```typescript
// 1. Submit all answers
{
  type: 'submit_answers',
  data: {
    answers: Record<number, string>,
    session_id: string
  }
}
```

---

## 7. EVENT HANDLERS

### 7.1 GraphViewer Events

| Event | Handler | Purpose | Parameters |
|-------|---------|---------|------------|
| `onNodeClick` | `handleNodeClick` | Select node, show detail panel | `nodeId: string` |
| `onNodeHover` | `handleNodeHover` | Show tooltip | `nodeId: string \| null` |
| `onNodeDoubleClick` | `handleNodeDoubleClick` | Expand node subgraph | `nodeId: string` |
| `onBackgroundClick` | `handleBackgroundClick` | Deselect node | - |
| `onZoom` | `handleZoom` | Update zoom level | `zoomLevel: number` |

**Implementation Example:**

```typescript
const handleNodeClick = (nodeId: string) => {
  graphStore.setSelectedNode(nodeId);
  // Optionally: fetch node details, expand ancestors/descendants
};

const handleNodeHover = (nodeId: string | null) => {
  graphStore.setHoveredNode(nodeId);
};
```

### 7.2 DialecticChat Events

| Event | Handler | Purpose | Parameters |
|-------|---------|---------|------------|
| `onAcceptSuggested` | `handleAcceptSuggested` | Auto-fill answer with suggested | `index: number, answer: string` |
| `onProvideOwn` | `handleProvideOwn` | Switch to custom input mode | `index: number` |
| `onAnswerChange` | `handleAnswerChange` | Update answer in store | `index: number, answer: string` |
| `onSubmit` | `handleSubmit` | Submit all answers to backend | - |

**Implementation Example:**

```typescript
const handleSubmit = async () => {
  const { answers } = dialecticStore;
  const payload: ClarificationResponse = {
    answers: Object.fromEntries(answers),
    session_id: sessionId,
  };

  dialecticStore.setPendingSubmit(true);

  try {
    // Send via WebSocket or REST
    dialecticWS.send(JSON.stringify({
      type: 'submit_answers',
      data: payload,
    }));
  } catch (error) {
    console.error('Failed to submit answers', error);
  } finally {
    dialecticStore.setPendingSubmit(false);
  }
};
```

### 7.3 Layout Events

| Event | Handler | Purpose | Parameters |
|-------|---------|---------|------------|
| `onDividerDrag` | `handleDividerDrag` | Resize split panes | `newRatio: number` |
| `onThemeToggle` | `handleThemeToggle` | Switch dark/light mode | - |
| `onMetricsToggle` | `handleMetricsToggle` | Collapse/expand metrics | - |
| `onLegendToggle` | `handleLegendToggle` | Collapse/expand legend | - |

---

## 8. IMPLEMENTATION ORDER

### Phase 1: Foundation (Week 1)

**Priority:** Setup, core infrastructure, static rendering

1. **Project Setup**
   - Initialize Vite + React + TypeScript
   - Install dependencies: `cosmograph`, `zustand`, `tailwindcss`, `@apache-arrow/es`
   - Configure TailwindCSS with dark mode
   - Create folder structure

2. **Type Definitions**
   - `src/types/graph.types.ts` (VizNode, VizEdge, GraphSnapshot, GraphDelta)
   - `src/types/dialectic.types.ts` (AmbiguityMarker, ClarificationQuestion)
   - `src/types/websocket.types.ts` (WebSocket message types)

3. **State Management**
   - `src/stores/graphStore.ts` (Zustand graph state)
   - `src/stores/uiStore.ts` (UI preferences)
   - Test with mock data

4. **Basic Layout**
   - `src/components/Layout.tsx` (Split-pane wrapper)
   - `src/App.tsx` (Root component)
   - Static header with theme toggle

**Deliverable:** App shell renders with split panes, theme toggle works

---

### Phase 2: Graph Visualization (Week 2)

**Priority:** Cosmograph integration, static snapshot rendering

5. **GraphViewer Component**
   - `src/components/GraphViewer.tsx`
   - Integrate Cosmograph
   - Render nodes/edges from snapshot
   - Basic node click/hover detection

6. **REST API Integration**
   - `src/utils/apiClient.ts` (HTTP client)
   - `src/hooks/useGraphSnapshot.ts` (Fetch initial snapshot)
   - Connect GraphViewer to real backend

7. **NodeTooltip Component**
   - `src/components/NodeTooltip.tsx`
   - Position near hovered node
   - Display node metadata
   - Connect to `hoveredNodeId` from store

8. **Legend Component**
   - `src/components/Legend.tsx`
   - Display node/edge color mappings
   - Sync with color mode (type vs status)

**Deliverable:** Graph renders from backend, tooltips work, legend displays

---

### Phase 3: Real-Time Updates (Week 3)

**Priority:** WebSocket integration, delta application

9. **Graph WebSocket Hook**
   - `src/hooks/useGraphWebSocket.ts`
   - Connect to `/api/viz/ws`
   - Handle snapshot, delta, heartbeat messages
   - Automatic reconnection with exponential backoff

10. **Delta Application Logic**
    - Update `graphStore.applyDelta()` implementation
    - Test with incremental updates
    - Ensure no re-renders of unchanged nodes

11. **Metrics Dashboard**
    - `src/components/MetricsDashboard.tsx`
    - Display live node/edge counts
    - Type/status breakdowns
    - Collapsible panel

**Deliverable:** Real-time graph updates work, metrics update live

---

### Phase 4: Dialectic Interface (Week 4)

**Priority:** Chat UI, ambiguity display, answer collection

12. **Dialectic Store**
    - `src/stores/dialecticStore.ts` (Zustand dialectic state)
    - Actions for setting answers, accepting suggested

13. **AmbiguityCard Component**
    - `src/components/AmbiguityCard.tsx`
    - Display category, text, question
    - Accept/Provide buttons

14. **QuestionCard Component**
    - `src/components/QuestionCard.tsx`
    - Suggested mode vs custom input mode
    - Answer validation

15. **DialecticChat Component**
    - `src/components/DialecticChat.tsx`
    - Phase indicator
    - List of ambiguities/questions
    - Submit button

**Deliverable:** Full dialectic UI, can collect and display answers

---

### Phase 5: Dialectic WebSocket (Week 5)

**Priority:** Real-time dialectic phase, backend integration

**NOTE:** This phase depends on backend implementation of `/api/dialectic/ws` endpoint, which does NOT exist yet. This is a future enhancement.

16. **Dialectic WebSocket Hook**
    - `src/hooks/useDialecticWebSocket.ts`
    - Connect to `/api/dialectic/ws` (to be implemented)
    - Handle phase changes, ambiguities, questions

17. **Submit Answers Integration**
    - Send answers via WebSocket
    - Handle submission confirmation
    - Transition to next phase

**Deliverable:** Real-time dialectic phase works end-to-end

---

### Phase 6: Polish & Accessibility (Week 6)

**Priority:** UX refinements, keyboard navigation, dark mode

18. **Keyboard Navigation**
    - Tab through nodes
    - Arrow keys to navigate
    - Enter to select
    - Escape to deselect

19. **Accessibility Audit**
    - Add ARIA labels
    - Test with screen reader
    - Ensure WCAG AA contrast
    - Focus indicators

20. **Performance Optimization**
    - Memoize expensive renders
    - Virtualize long lists in DialecticChat
    - Debounce hover events
    - Profile with 10K+ nodes

21. **Error Handling**
    - WebSocket disconnect handling
    - Retry logic with exponential backoff
    - User-friendly error messages
    - Offline indicator

**Deliverable:** Production-ready, accessible, performant UI

---

### Phase 7: Advanced Features (Week 7+)

**Priority:** Nice-to-haves, developer experience

22. **Node Detail Panel**
    - Click node to open detail drawer
    - Show full content, metadata
    - Link to related nodes

23. **Search & Filter**
    - Search nodes by label
    - Filter by type/status
    - Highlight matching nodes

24. **Timeline Scrubbing**
    - Scrub through graph history
    - Play/pause animation
    - Jump to specific timestamp

25. **Export Functionality**
    - Export graph as PNG/SVG
    - Export as GraphML
    - Copy node details

**Deliverable:** Advanced features for power users

---

## 9. DEPENDENCY GRAPH

```
Project Setup (1)
  └─> Type Definitions (2)
       └─> State Management (3)
            ├─> Basic Layout (4)
            │    └─> GraphViewer (5)
            │         ├─> REST API Integration (6)
            │         │    └─> NodeTooltip (7)
            │         │         └─> Legend (8)
            │         │              └─> Graph WebSocket Hook (9)
            │         │                   └─> Delta Application (10)
            │         │                        └─> Metrics Dashboard (11)
            │         └─> [Real-time graph complete]
            │
            └─> Dialectic Store (12)
                 └─> AmbiguityCard (13)
                      └─> QuestionCard (14)
                           └─> DialecticChat (15)
                                └─> Dialectic WebSocket Hook (16)
                                     └─> Submit Integration (17)
                                          └─> [Dialectic complete]
                                               └─> Polish & A11y (18-21)
                                                    └─> Advanced Features (22-25)
```

---

## 10. TESTING STRATEGY

### 10.1 Unit Tests (Vitest + React Testing Library)

**Coverage Targets:** 80% line coverage

| Component | Test Cases |
|-----------|------------|
| `graphStore` | setSnapshot, applyDelta, sequence ordering |
| `dialecticStore` | setAnswer, acceptSuggested, clearAnswer |
| `NodeTooltip` | renders correctly, positions near node, shows all fields |
| `AmbiguityCard` | accept suggested, provide own, state transitions |
| `QuestionCard` | custom input, validation, submission |

### 10.2 Integration Tests (Playwright)

| Scenario | Steps |
|----------|-------|
| Graph Load | Connect to backend, render snapshot, verify node count |
| Real-time Update | Trigger delta, verify nodes update without full re-render |
| Node Interaction | Click node, verify selection, hover shows tooltip |
| Dialectic Flow | Display ambiguities, accept suggested, submit answers |

### 10.3 E2E Tests (Playwright)

| Scenario | Steps |
|----------|-------|
| Full TDD Cycle | Start session, answer dialectic questions, watch graph build |
| WebSocket Reconnect | Disconnect backend, verify reconnect, verify no data loss |
| Color Mode Switch | Toggle type/status, verify colors update |

---

## 11. ENVIRONMENT VARIABLES

**File:** `.env`

```bash
# API Configuration
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000

# Feature Flags
VITE_ENABLE_DIALECTIC_WS=false  # Set true when backend implements /api/dialectic/ws
VITE_ENABLE_ARROW_IPC=true      # Use Arrow IPC format for large graphs
VITE_ENABLE_TIMELINE=false      # Future: timeline scrubbing

# Performance
VITE_MAX_NODES_WARNING=5000     # Warn user if graph exceeds this
VITE_ENABLE_DEBUG_LOGGING=false

# UI Defaults
VITE_DEFAULT_THEME=dark
VITE_DEFAULT_COLOR_MODE=type
```

---

## 12. NPM DEPENDENCIES

**File:** `package.json`

```json
{
  "name": "paragon-ui",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "test:e2e": "playwright test",
    "lint": "eslint . --ext ts,tsx",
    "format": "prettier --write \"src/**/*.{ts,tsx}\""
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "cosmograph": "^1.5.0",
    "zustand": "^4.5.0",
    "@apache-arrow/es": "^14.0.2",
    "react-resizable-panels": "^2.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.5.0",
    "vite": "^5.4.0",
    "vitest": "^2.0.0",
    "@testing-library/react": "^16.0.0",
    "@testing-library/jest-dom": "^6.5.0",
    "@playwright/test": "^1.47.0",
    "tailwindcss": "^3.4.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "eslint": "^9.0.0",
    "prettier": "^3.3.0"
  }
}
```

---

## 13. SUCCESS CRITERIA

### 13.1 Performance

- [ ] Render 10K nodes in <5 seconds
- [ ] Maintain 60 FPS during pan/zoom
- [ ] Real-time delta updates <100ms latency
- [ ] Tooltip hover response <16ms
- [ ] Bundle size <500KB (gzipped)

### 13.2 Functionality

- [ ] Graph renders from backend snapshot
- [ ] Real-time deltas apply correctly
- [ ] Node click/hover interactions work
- [ ] Dialectic ambiguities display
- [ ] Answer submission works
- [ ] WebSocket reconnects automatically
- [ ] Color mode switching works
- [ ] Theme toggle works

### 13.3 Accessibility

- [ ] Keyboard-only navigation functional
- [ ] WCAG AA contrast ratios
- [ ] Screen reader compatible
- [ ] Focus indicators visible
- [ ] No accessibility errors in Lighthouse

### 13.4 Quality

- [ ] 80% unit test coverage
- [ ] All E2E tests passing
- [ ] No TypeScript errors
- [ ] No ESLint warnings
- [ ] All components documented

---

## 14. FUTURE ENHANCEMENTS

### 14.1 Phase 8: Timeline Debugging

- Integrate with Rerun.io backend
- Scrub through graph history
- Replay node/edge mutations
- Compare snapshots at different times

### 14.2 Phase 9: Developer Tools

- Graph diff viewer (compare two versions)
- Export to GraphML/DOT format
- Node search with regex
- Custom layout algorithms

### 14.3 Phase 10: Collaboration

- Multi-user cursor presence
- Shared annotations on nodes
- Real-time chat sidebar
- Session replay for demos

---

## REFERENCES

### Backend API
- `/Users/lauferva/paragon/api/routes.py` - REST and WebSocket endpoints
- `/Users/lauferva/paragon/viz/core.py` - Data models (VizNode, VizEdge, GraphSnapshot, GraphDelta)

### Backend Schemas
- `/Users/lauferva/paragon/agents/schemas.py` - TypeScript type reference (AmbiguityMarker, ClarificationQuestion)
- `/Users/lauferva/paragon/core/ontology.py` - NodeType, NodeStatus, EdgeType enums

### Research
- `/Users/lauferva/paragon/docs/RESEARCH_RT_VISUALIZATION.md` - Technology stack research
- `/Users/lauferva/paragon/docs/RESEARCH_ADAPTIVE_QUESTIONING.md` - Dialectic system design

### External
- [Cosmograph Documentation](https://github.com/cosmosgl/graph)
- [Zustand Documentation](https://github.com/pmndrs/zustand)
- [Apache Arrow JS](https://arrow.apache.org/docs/js/)
- [React 18 Documentation](https://react.dev/)

---

**End of Specification**

**Version:** 1.0.0
**Last Updated:** 2025-12-06
**Author:** Claude Sonnet 4.5
**Status:** READY FOR IMPLEMENTATION
