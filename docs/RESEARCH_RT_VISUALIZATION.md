# RESEARCH REPORT: Real-Time Graph Visualization & Dialectic GUI for Project Paragon

**Date:** 2025-12-06
**Research Agent:** Claude Sonnet 4.5
**Status:** Complete

---

## EXECUTIVE SUMMARY

This report evaluates technology stacks and architectural patterns for implementing a real-time graph visualization and interactive dialectic GUI for Project Paragon. The system must support 10K+ node graphs with smooth real-time updates via WebSocket, timeline scrubbing for temporal debugging, and an intuitive question-answer interface for the dialectic research phase.

**Key Recommendation:** Hybrid architecture combining **Cosmograph** for graph rendering, **FastAPI WebSocket** backend with **Polars/Arrow IPC** serialization, and a **React** frontend with custom dialectic chat UI.

---

## 1. TECHNOLOGY RECOMMENDATION

### 1.1 Frontend Graph Visualization: **Cosmograph**

**Rationale:**
- **GPU-Accelerated Performance:** Cosmograph uses WebGL for both layout computation and rendering, making it the fastest option for 10K+ nodes
- **Benchmark Data:** Renders 10k nodes in 5-10 seconds vs. 27s for D3-based solutions and 10.5 minutes for Cytoscape
- **Scalability:** Handles hundreds of thousands of nodes in browser without degradation
- **Modern Architecture:** Built for 2025 web standards with native WebAssembly optimization

**Alternatives Considered:**
- **Sigma.js:** Strong WebGL rendering (100k edges easily) but force-directed layout struggles beyond 50k edges. Good fallback option.
- **React Force Graph:** Well-suited for React apps with WebGL/Canvas support, but doesn't match Cosmograph's GPU-accelerated layout.

### 1.2 Backend: **FastAPI + WebSocket**

**Rationale:**
- **Native Async Support:** FastAPI's built-in async/await aligns with Paragon's `asyncio` architecture
- **Production-Ready WebSocket:** Provides `WebSocket` class from Starlette with automatic upgrade handling
- **Low Latency:** WebSocket enables 1M+ data points/second with good network, 30k/sec with poor network
- **Integration:** Works seamlessly with Granian runtime (specified in `/config/paragon.toml`)

### 1.3 Data Serialization: **Polars + Apache Arrow IPC**

**Rationale:**
- **Zero-Copy Transfer:** Arrow IPC format eliminates deserialization overhead between WebAssembly and JavaScript
- **Existing Integration:** Paragon already uses Polars (see `/viz/core.py:serialize_to_arrow`)
- **Performance:** 3-10x faster than JSON for large graphs due to columnar format

### 1.4 Frontend Framework: **React + TypeScript**

**Rationale:**
- **Component-Based Architecture:** Natural fit for dialectic chat interface + graph viewer split pane
- **Ecosystem:** Rich ecosystem of WebSocket hooks, UI libraries (shadcn/ui for dark mode)

---

## 2. ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  React Frontend (TypeScript)                             │   │
│  │  ┌──────────────────┐  ┌────────────────────────────┐   │   │
│  │  │ Dialectic Chat   │  │  Graph Viewer (Cosmograph) │   │   │
│  │  │ Component        │  │  - Node hover tooltips     │   │   │
│  │  │ - Question list  │  │  - Timeline scrubber       │   │   │
│  │  │ - Suggested ans  │  │  - Dark mode support       │   │   │
│  │  │ - User input     │  │  - Accessibility (ARIA)    │   │   │
│  │  └────────┬─────────┘  └────────┬───────────────────┘   │   │
│  │           │    WebSocket         │  WebSocket             │   │
│  │           │    /ws/dialectic     │  /ws/graph             │   │
│  └───────────┼──────────────────────┼────────────────────────┘   │
└──────────────┼──────────────────────┼────────────────────────────┘
               │                      │
               ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              FASTAPI BACKEND (Granian Runtime)                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  WebSocket Endpoints                                      │   │
│  │  ┌────────────────┐  ┌─────────────────────────────┐     │   │
│  │  │ /ws/dialectic  │  │  /ws/graph                  │     │   │
│  │  │ Handler        │  │  Handler                    │     │   │
│  │  └───────┬────────┘  └────────┬────────────────────┘     │   │
│  │          │                     │                          │   │
│  │          ▼                     ▼                          │   │
│  │  ┌────────────────┐  ┌─────────────────────────────┐     │   │
│  │  │ Orchestrator   │  │  VizGraph (viz/core.py)     │     │   │
│  │  │ dialectic_node │◄─┤  - GraphSnapshot            │     │   │
│  │  │ clarification  │  │  - GraphDelta (incremental) │     │   │
│  │  │ research_node  │  │  - Arrow IPC serialization  │     │   │
│  │  └───────┬────────┘  └─────────┬───────────────────┘     │   │
│  │          │                     │                          │   │
│  │          ▼                     ▼                          │   │
│  │  ┌────────────────────────────────────────────────┐       │   │
│  │  │      ParagonDB (core/graph_db.py)              │       │   │
│  │  │      rustworkx PyDiGraph backend               │       │   │
│  │  └────────────────────────────────────────────────┘       │   │
│  │                                                            │   │
│  │  ┌────────────────────────────────────────────────┐       │   │
│  │  │  RerunLogger (infrastructure/rerun_logger.py)  │       │   │
│  │  │  - Timeline recording (.rrd files)             │       │   │
│  │  └────────────────────────────────────────────────┘       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. INTEGRATION POINTS

### 3.1 Files Requiring Modification

| File | Change |
|------|--------|
| **NEW: `api/websocket_graph.py`** | WebSocket endpoint for graph visualization updates |
| **NEW: `api/websocket_dialectic.py`** | WebSocket endpoint for dialectic chat interface |
| **MODIFY: `viz/core.py`** | Add `create_delta_from_mutation()` function |
| **MODIFY: `core/graph_db.py`** | Add mutation callbacks for real-time updates |
| **NEW: `api/main.py`** | FastAPI application entry point |

### 3.2 Frontend File Structure (New Repository)

```
paragon-ui/
├── src/
│   ├── components/
│   │   ├── GraphViewer.tsx          # Cosmograph integration
│   │   ├── DialecticChat.tsx        # Question/answer interface
│   │   ├── NodeTooltip.tsx          # Hover info popup
│   │   ├── MetricsDashboard.tsx     # From infrastructure/metrics.py
│   │   └── Legend.tsx               # Node/edge type legend
│   ├── hooks/
│   │   ├── useGraphWebSocket.ts
│   │   └── useDialecticWebSocket.ts
│   └── stores/
│       └── graphStore.ts            # Zustand state management
```

---

## 4. IMPLEMENTATION PRIORITY

### Phase 1: Core Graph Visualization (Week 1-2)
- Create FastAPI app with `/api/snapshot` endpoint
- Initialize React project with Cosmograph
- Implement node hover tooltips

### Phase 2: Real-Time WebSocket Updates (Week 3)
- Add mutation callbacks to ParagonDB
- Implement GraphDelta broadcasting
- Create `useGraphWebSocket` hook

### Phase 3: Dialectic Chat Interface (Week 4)
- Create WebSocket dialectic endpoint
- Build DialecticChat component
- Integrate with orchestrator

### Phase 4: Advanced Features (Week 5-6)
- Click-to-expand node detail panel
- Metrics dashboard integration
- Layout optimization

### Phase 5: Dynamic Headers & Unified Report (Week 7)
- Context-aware header generation
- Combined topology + metrics + legend view

---

## 5. GUI COMPONENT DESIGNS

### 5.1 Node Hover Tooltip

```
┌─────────────────────────────────────┐
│ CODE: calculate_hash                │  ← Larger font (16px)
│ (Function in crypto/hash.py)        │  ← Smaller font (12px), gray
├─────────────────────────────────────┤
│ Status: VERIFIED ✓                  │
│ Created: 2025-12-06 14:32           │
│ Agent: builder_agent_1              │
├─────────────────────────────────────┤
│ Traces to: REQ-8e6243b6             │
│ Implements: SPEC-a1b2c3d4           │
├─────────────────────────────────────┤
│ Click for details                   │
└─────────────────────────────────────┘
```

### 5.2 Dialectic Chat Interface

```
┌─────────────────────────────────────────────────────┐
│  Ambiguity Analysis                     [Phase: 2/5] │
├─────────────────────────────────────────────────────┤
│  Found 3 ambiguities in your specification:         │
│                                                      │
│  ┌────────────────────────────────────────────────┐ │
│  │ 🟡 SUBJECTIVE: "fast sorting function"        │ │
│  │                                                │ │
│  │ Question: What performance target?             │ │
│  │                                                │ │
│  │ Suggested: O(n log n), 1M elements in <1s     │ │
│  │                                                │ │
│  │ [ Accept Suggested ]  [ Provide Own Answer ]  │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 5.3 Rollback UX

When user says "That approach won't work":
1. Show branch point visualization
2. Dim nodes created after branch point
3. Confirm removal before executing
4. Animate nodes fading out

---

## 6. RISK ASSESSMENT

| Risk | Severity | Mitigation |
|------|----------|------------|
| WebSocket Connection Stability | HIGH | Exponential backoff retry, resync on reconnect |
| Browser Performance (10K+ nodes) | MEDIUM | Cosmograph GPU acceleration, LOD rendering |
| Arrow IPC Browser Compatibility | LOW | Fallback to JSON for older browsers |
| Dialectic State Desync | MEDIUM | Sequence numbers, periodic full state sync |
| Dark Mode Accessibility | LOW | WCAG AA contrast testing |

---

## 7. SUCCESS CRITERIA

- [ ] Render 10k+ nodes at 60 FPS
- [ ] Real-time updates with <100ms latency
- [ ] Dialectic questions appear within 1 second
- [ ] Tooltip hover response <16ms
- [ ] Dark mode passes WCAG AA contrast
- [ ] Keyboard-only navigation functional

---

## SOURCES

- [Cosmograph GPU-accelerated graph](https://github.com/cosmosgl/graph)
- [FastAPI WebSockets docs](https://fastapi.tiangolo.com/advanced/websockets/)
- [Apache Arrow IPC streaming](https://arrow.apache.org/docs/python/ipc.html)
- [Dark mode UI best practices 2025](https://www.designstudiouiux.com/blog/dark-mode-ui-design-best-practices/)
