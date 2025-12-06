# Paragon UI

Frontend application for the Paragon graph-native development platform.

## Features

- **Real-time Graph Visualization**: Live updates via WebSocket
- **Dialectic Mode**: Interactive requirement clarification
- **Type-Safe**: Full TypeScript implementation
- **State Management**: Zustand stores for predictable state
- **Modern UI**: TailwindCSS with dark mode support
- **Responsive**: Resizable panels and adaptive layouts

## Project Structure

```
paragon-ui/
├── src/
│   ├── types/          # TypeScript type definitions
│   │   ├── graph.types.ts
│   │   ├── dialectic.types.ts
│   │   └── websocket.types.ts
│   ├── stores/         # Zustand state stores
│   │   ├── graphStore.ts
│   │   ├── dialecticStore.ts
│   │   └── uiStore.ts
│   ├── styles/         # Global styles
│   │   └── globals.css
│   ├── App.tsx         # Root component
│   └── main.tsx        # Entry point
├── public/             # Static assets
├── index.html          # HTML template
├── package.json        # Dependencies
├── tsconfig.json       # TypeScript config
├── vite.config.ts      # Vite config
└── tailwind.config.js  # TailwindCSS config
```

## Getting Started

### Install Dependencies

```bash
npm install
```

### Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build for Production

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

### Environment Variables

- `VITE_API_URL`: Backend API URL (default: http://localhost:8000)
- `VITE_GRAPH_WS_URL`: Graph WebSocket URL
- `VITE_DIALECTIC_WS_URL`: Dialectic WebSocket URL

## State Management

### Graph Store (`graphStore.ts`)

Manages graph data, WebSocket connection, node selection, filtering, and history.

```typescript
const { nodes, edges, selectNode, applyDelta } = useGraphStore();
```

### Dialectic Store (`dialecticStore.ts`)

Manages dialectic sessions, questions, answers, and clarification flow.

```typescript
const { state, submitAnswer, getCurrentQuestion } = useDialecticStore();
```

### UI Store (`uiStore.ts`)

Manages UI state, theme, layout, panels, and notifications.

```typescript
const { theme, setTheme, addNotification } = useUIStore();
```

## Type Definitions

All types are fully documented in `src/types/`:

- **graph.types.ts**: Graph nodes, edges, snapshots, deltas
- **dialectic.types.ts**: Ambiguities, questions, answers, sessions
- **websocket.types.ts**: WebSocket message types and connection state

## Next Steps

The foundation is complete. Next phase will add:

1. Graph visualization components (react-force-graph or cytoscape.js)
2. Dialectic UI components (question flow, ambiguity markers)
3. Inspector panel (node details, relationships)
4. Timeline and metrics views
5. WebSocket integration testing

## Tech Stack

- **React 18**: UI framework
- **TypeScript**: Type safety
- **Vite**: Build tool
- **Zustand**: State management
- **TailwindCSS**: Styling
- **react-resizable-panels**: Layout management
