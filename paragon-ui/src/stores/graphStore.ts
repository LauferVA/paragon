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
    delta.edges_removed.forEach(({ source, target }) => newEdges.delete(`${source}-${target}`));

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
