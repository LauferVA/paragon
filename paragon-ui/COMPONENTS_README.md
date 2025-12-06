# Paragon UI Components

**Status:** IMPLEMENTED - Ready for Integration
**Created:** 2025-12-06
**Location:** `/Users/lauferva/paragon/paragon-ui/src/components/`

---

## Overview

All 8 React components specified in FRONTEND_SPEC.md have been implemented and are ready for integration once the foundation (stores, types, hooks) is in place.

---

## Components Implemented

### 1. Layout.tsx (2.4 KB)
**Responsibility:** Main application layout with split-pane design

**Features:**
- Header with app name, version, theme toggle, settings button
- Resizable split panes using `react-resizable-panels`
- Dark/light theme support via Zustand store
- Auto-saves layout preferences

**Dependencies:**
- `react-resizable-panels`
- `stores/uiStore` (theme, splitRatio)
- `GraphViewer` and `DialecticChat` components

**Props:** Optional `sessionId` for dialectic session

---

### 2. GraphViewer.tsx (4.5 KB)
**Responsibility:** Main graph visualization component

**Features:**
- Placeholder for Cosmograph integration (marked with TODO comments)
- Connection status indicator
- Handles node click/hover events
- Empty state display when no graph data
- Integrates NodeTooltip, Legend, and MetricsDashboard as overlays

**Dependencies:**
- `stores/graphStore` (nodes, edges, snapshot, selectedNodeId, hoveredNodeId)
- `NodeTooltip`, `Legend`, `MetricsDashboard` components

**Props:**
- `width`, `height` (optional)
- `colorMode` ('type' | 'status', default: 'type')
- `onNodeClick`, `onNodeHover` callbacks

**Note:** Cosmograph integration is marked as TODO. The canvas div is prepared for mounting.

---

### 3. NodeTooltip.tsx (2.9 KB)
**Responsibility:** Displays detailed node information on hover

**Features:**
- Fixed position near cursor with offset
- Shows node label, type, status badge
- Displays metadata (created_at, created_by, layer, teleology_status)
- Root/Leaf indicators
- Status-based color coding (VERIFIED=green, PENDING=yellow, etc.)
- Fade-in animation

**Dependencies:**
- `types/graph.types` (VizNode interface)

**Props:**
- `node: VizNode | null`
- `position: { x: number; y: number }`
- `visible: boolean`

---

### 4. MetricsDashboard.tsx (3.5 KB)
**Responsibility:** Displays live graph statistics

**Features:**
- Collapsible panel with toggle button
- Core metrics: Nodes, Edges, Layers, Cycle status
- Additional metrics: Root count, Leaf count, Version
- Color-coded metric cards (blue, teal, purple, green/red)
- Last updated timestamp

**Dependencies:**
- `types/graph.types` (GraphSnapshot interface)
- `stores/uiStore` (metricsCollapsed, toggleMetrics)

**Props:**
- `snapshot: GraphSnapshot | null`
- `collapsed?: boolean` (optional override)
- `onToggle?: () => void` (optional callback)

**Helper Components:**
- `MetricCard` - Individual metric display
- `MetricRow` - Key-value row

---

### 5. Legend.tsx (3.7 KB)
**Responsibility:** Color legend for node/edge types

**Features:**
- Collapsible panel
- Node type colors (REQ=red, SPEC=orange, CODE=teal, TEST=dark blue, DOC=medium blue)
- Node status colors (VERIFIED=green, PENDING=yellow, FAILED=red, DRAFT=gray)
- Edge type colors (IMPLEMENTS=teal, TRACES_TO=red, DEPENDS_ON=gray, VALIDATES=blue)
- Switches between type/status mode based on `colorMode` prop

**Dependencies:**
- `stores/uiStore` (legendCollapsed, toggleLegend)

**Props:**
- `colorMode?: 'type' | 'status'` (default: 'type')
- `collapsed?: boolean` (optional override)
- `onToggle?: () => void` (optional callback)

**Color Mappings:** Matches backend viz/core.py definitions

---

### 6. DialecticChat.tsx (6.4 KB)
**Responsibility:** Interactive chat interface for dialectic phase

**Features:**
- Phase indicator (DIALECTIC, CLARIFICATION, RESEARCH, IDLE)
- Connection status indicator
- Auto-scroll to latest message
- Lists AmbiguityCards or QuestionCards based on phase
- Submit button with answer count tracking
- Empty states for each phase
- Loading states with icons and messages

**Dependencies:**
- `stores/dialecticStore` (phase, ambiguities, questions, answers)
- `AmbiguityCard`, `QuestionCard` components

**Props:**
- `sessionId: string` (required)
- `onComplete?: () => void` (optional callback)

**Helper Components:**
- `PhaseIndicator` - Badge showing current phase with icon

**Event Handlers:**
- `handleAcceptSuggested` - Auto-fill with suggested answer
- `handleProvideOwn` - Switch to custom input mode
- `handleAnswerChange` - Update answer in store
- `handleSubmit` - Submit all answers (TODO: WebSocket/REST integration)

---

### 7. AmbiguityCard.tsx (4.3 KB)
**Responsibility:** Displays single detected ambiguity

**Features:**
- Category badge with icon and color coding
- BLOCKING impact highlighted in red
- Shows ambiguous text with yellow highlight
- Displays clarification question
- Shows suggested answer in blue box
- Shows user's answer in green box when answered
- Two action buttons: "Accept Suggested" and "Provide Own Answer"
- Answered state indicator

**Dependencies:**
- `types/dialectic.types` (AmbiguityMarker interface)

**Props:**
- `ambiguity: AmbiguityMarker`
- `index: number`
- `onAcceptSuggested: (index: number, answer: string) => void`
- `onProvidOwn: (index: number) => void`
- `answer?: string`

**Category Icons:**
- BLOCKING: 🔴
- SUBJECTIVE: 🟡
- COMPARATIVE: 🟠
- UNDEFINED_PRONOUN: 🔵
- UNDEFINED_TERM: 🟣
- MISSING_CONTEXT: 🟢

---

### 8. QuestionCard.tsx (5.1 KB)
**Responsibility:** Interactive card for collecting clarification answers

**Features:**
- Priority badge (high=red, medium=yellow, low=blue)
- Mode toggle: Suggested Options vs Custom Answer
- Suggested mode: List of clickable options
- Custom mode: Textarea input with submit button
- Answer validation (non-empty)
- Answered state display in green box
- Context display (if provided)

**Dependencies:**
- `types/dialectic.types` (ClarificationQuestion interface)

**Props:**
- `question: ClarificationQuestion`
- `index: number`
- `answer?: string`
- `onAnswerChange: (index: number, answer: string) => void`
- `disabled?: boolean`

**States:**
- `mode: 'suggested' | 'custom'` - Toggle between modes
- `customAnswer: string` - Local state for custom input

---

## Styling

All components use **TailwindCSS** with the following conventions:

### Color Palette (Dark Mode Native)
- Background: `bg-gray-900` (main), `bg-gray-800` (cards/panels)
- Borders: `border-gray-700`
- Text: `text-white` (primary), `text-gray-400` (secondary), `text-gray-500` (tertiary)
- Accents:
  - Blue: `bg-blue-600` (primary action)
  - Green: Success states
  - Red: Error/blocking states
  - Yellow: Warning/subjective states
  - Teal: Code/implementation
  - Purple: Research/specs

### Component Classes
- Cards: `bg-gray-800 border border-gray-700 rounded-lg p-4`
- Buttons (primary): `bg-blue-600 hover:bg-blue-500 text-white`
- Buttons (secondary): `bg-gray-700 hover:bg-gray-600 text-white`
- Badges: `px-3 py-1 rounded-full text-xs font-medium border`

### Responsive Design
- All components use flexbox/grid for layout
- Layout.tsx uses `react-resizable-panels` for split panes
- Mobile stacking planned but not yet implemented

---

## Dependencies Required

These dependencies must be installed for the components to work:

```json
{
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "zustand": "^4.5.0",
    "react-resizable-panels": "^2.0.0"
  }
}
```

**Optional (for full functionality):**
- `cosmograph`: "^1.5.0" (for GraphViewer)
- `@apache-arrow/es`: "^14.0.2" (for Arrow IPC data)

---

## Foundation Files Needed

Before these components can be used, the following foundation files must be created:

### 1. Type Definitions
- `/src/types/graph.types.ts` - VizNode, VizEdge, GraphSnapshot, GraphDelta
- `/src/types/dialectic.types.ts` - AmbiguityMarker, ClarificationQuestion
- `/src/types/websocket.types.ts` - WebSocket message types (optional)

### 2. Zustand Stores
- `/src/stores/graphStore.ts` - Graph state management
- `/src/stores/dialecticStore.ts` - Dialectic state management
- `/src/stores/uiStore.ts` - UI preferences (theme, layout)

### 3. Hooks (Optional but Recommended)
- `/src/hooks/useGraphWebSocket.ts` - WebSocket connection for graph updates
- `/src/hooks/useDialecticWebSocket.ts` - WebSocket for dialectic phase
- `/src/hooks/useGraphSnapshot.ts` - Initial snapshot fetch

### 4. Root Files
- `/src/App.tsx` - Root component (uses Layout)
- `/src/main.tsx` - Entry point
- `/src/styles/globals.css` - TailwindCSS imports

### 5. Configuration
- `/package.json` - Dependencies and scripts
- `/tsconfig.json` - TypeScript configuration
- `/vite.config.ts` - Vite build configuration
- `/tailwind.config.js` - TailwindCSS configuration

---

## Integration Checklist

- [ ] Install dependencies (`npm install` or `yarn install`)
- [ ] Create type definition files
- [ ] Implement Zustand stores
- [ ] Create hooks for WebSocket/API
- [ ] Set up TailwindCSS configuration
- [ ] Create App.tsx using Layout component
- [ ] Test with mock data
- [ ] Connect to backend API
- [ ] Integrate Cosmograph in GraphViewer
- [ ] Test real-time updates
- [ ] Add accessibility features (keyboard nav, ARIA labels)

---

## Known TODOs

### GraphViewer.tsx
- Line 33-39: Cosmograph initialization commented out
- Line 49-51: Cosmograph data update commented out

### DialecticChat.tsx
- Line 40: WebSocket/REST API integration for answer submission

### All Components
- Accessibility improvements (ARIA labels, keyboard navigation)
- Mobile responsive layout
- Error boundaries
- Loading states
- Animations/transitions

---

## Testing Recommendations

### Unit Tests (Vitest + React Testing Library)
1. **NodeTooltip**: Renders correctly, shows all fields, positions properly
2. **AmbiguityCard**: Accept/provide buttons work, state transitions
3. **QuestionCard**: Mode switching, custom input, validation
4. **MetricsDashboard**: Displays metrics correctly, collapse/expand works
5. **Legend**: Shows correct colors, switches modes

### Integration Tests (Playwright)
1. **Layout**: Renders both panels, theme toggle works, resize works
2. **GraphViewer**: Displays placeholder, shows connection status
3. **DialecticChat**: Phase indicator changes, cards render, submit works

---

## File Sizes

| Component | Size | Lines |
|-----------|------|-------|
| Layout.tsx | 2.4 KB | ~75 |
| GraphViewer.tsx | 4.5 KB | ~130 |
| NodeTooltip.tsx | 2.9 KB | ~95 |
| MetricsDashboard.tsx | 3.5 KB | ~115 |
| Legend.tsx | 3.7 KB | ~120 |
| DialecticChat.tsx | 6.4 KB | ~190 |
| AmbiguityCard.tsx | 4.3 KB | ~135 |
| QuestionCard.tsx | 5.1 KB | ~160 |
| **Total** | **32.8 KB** | **~1,020** |

---

## Next Steps

1. **Immediate:** Create foundation files (types, stores, hooks)
2. **Phase 1:** Set up Vite project with dependencies
3. **Phase 2:** Integrate components with mock data for testing
4. **Phase 3:** Connect to backend API
5. **Phase 4:** Integrate Cosmograph for graph visualization
6. **Phase 5:** Polish, accessibility, performance optimization

---

**Status:** All components ready and waiting for foundation setup.

**Contact:** Claude Sonnet 4.5
**Date:** 2025-12-06
