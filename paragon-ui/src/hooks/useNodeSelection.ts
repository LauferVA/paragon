import { useCallback } from 'react';
import { useGraphStore } from '../stores/graphStore';
import { VizNode } from '../types/graph.types';

export function useNodeSelection() {
  const {
    nodes,
    selectedNodeId,
    hoveredNodeId,
    setSelectedNode,
    setHoveredNode,
  } = useGraphStore();

  const selectedNode: VizNode | null = selectedNodeId
    ? nodes.get(selectedNodeId) ?? null
    : null;

  const hoveredNode: VizNode | null = hoveredNodeId
    ? nodes.get(hoveredNodeId) ?? null
    : null;

  const handleNodeClick = useCallback((nodeId: string) => {
    setSelectedNode(nodeId === selectedNodeId ? null : nodeId);
  }, [selectedNodeId, setSelectedNode]);

  const handleNodeHover = useCallback((nodeId: string | null) => {
    setHoveredNode(nodeId);
  }, [setHoveredNode]);

  const handleBackgroundClick = useCallback(() => {
    setSelectedNode(null);
  }, [setSelectedNode]);

  return {
    selectedNode,
    hoveredNode,
    selectedNodeId,
    hoveredNodeId,
    handleNodeClick,
    handleNodeHover,
    handleBackgroundClick,
  };
}
