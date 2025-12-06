import React, { useEffect, useRef, useMemo } from 'react';
import { useGraphStore } from '../stores/graphStore';
import { useUIStore } from '../stores/uiStore';
import { useNodeSelection } from '../hooks/useNodeSelection';
import { NodeTooltip } from './NodeTooltip';
import { Legend } from './Legend';
import { MetricsDashboard } from './MetricsDashboard';
import { VizNode } from '../types/graph.types';
import { NODE_TYPE_COLORS, NODE_STATUS_COLORS, EDGE_TYPE_COLORS } from '../utils/colorMaps';

export function GraphViewer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const { nodes, edges, snapshot, colorMode } = useGraphStore();
  const { theme } = useUIStore();
  const { hoveredNode, selectedNodeId, handleNodeClick, handleNodeHover, handleBackgroundClick } = useNodeSelection();

  const [tooltipPos, setTooltipPos] = React.useState({ x: 0, y: 0 });
  const [dimensions, setDimensions] = React.useState({ width: 800, height: 600 });

  const nodeArray = useMemo(() => Array.from(nodes.values()), [nodes]);
  const edgeArray = useMemo(() => Array.from(edges.values()), [edges]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = dimensions;
    canvas.width = width;
    canvas.height = height;

    ctx.fillStyle = theme === 'dark' ? '#111827' : '#ffffff';
    ctx.fillRect(0, 0, width, height);

    const nodePositions = new Map<string, { x: number; y: number }>();
    const layerMap = new Map<number, VizNode[]>();

    nodeArray.forEach((node) => {
      const layer = node.layer ?? 0;
      if (!layerMap.has(layer)) layerMap.set(layer, []);
      layerMap.get(layer)!.push(node);
    });

    const layers = Array.from(layerMap.keys()).sort((a, b) => a - b);
    const layerCount = layers.length || 1;
    const layerHeight = height / (layerCount + 1);

    layers.forEach((layer, layerIndex) => {
      const nodesInLayer = layerMap.get(layer)!;
      const nodeWidth = width / (nodesInLayer.length + 1);
      nodesInLayer.forEach((node, nodeIndex) => {
        nodePositions.set(node.id, {
          x: node.x ?? nodeWidth * (nodeIndex + 1),
          y: node.y ?? layerHeight * (layerIndex + 1),
        });
      });
    });

    ctx.lineWidth = 1;
    edgeArray.forEach((edge) => {
      const sourcePos = nodePositions.get(edge.source);
      const targetPos = nodePositions.get(edge.target);
      if (!sourcePos || !targetPos) return;

      ctx.strokeStyle = EDGE_TYPE_COLORS[edge.type] || EDGE_TYPE_COLORS.DEFAULT;
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      ctx.moveTo(sourcePos.x, sourcePos.y);
      ctx.lineTo(targetPos.x, targetPos.y);
      ctx.stroke();
    });
    ctx.globalAlpha = 1;

    nodeArray.forEach((node) => {
      const pos = nodePositions.get(node.id);
      if (!pos) return;

      const isSelected = node.id === selectedNodeId;
      const radius = (node.size ?? 8) * (isSelected ? 1.5 : 1);

      const color = colorMode === 'status'
        ? NODE_STATUS_COLORS[node.status] || NODE_STATUS_COLORS.DEFAULT
        : NODE_TYPE_COLORS[node.type] || NODE_TYPE_COLORS.DEFAULT;

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      if (isSelected) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      ctx.fillStyle = theme === 'dark' ? '#ffffff' : '#000000';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(node.label.slice(0, 12), pos.x, pos.y + radius + 12);
    });
  }, [nodeArray, edgeArray, dimensions, theme, colorMode, selectedNodeId]);

  const handleMouseMove = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    let foundNode: VizNode | null = null;
    const layers = new Map<number, VizNode[]>();
    nodeArray.forEach((node) => {
      const layer = node.layer ?? 0;
      if (!layers.has(layer)) layers.set(layer, []);
      layers.get(layer)!.push(node);
    });

    const sortedLayers = Array.from(layers.keys()).sort((a, b) => a - b);
    const layerHeight = dimensions.height / (sortedLayers.length + 1);

    sortedLayers.forEach((layer, layerIndex) => {
      const nodesInLayer = layers.get(layer)!;
      const nodeWidth = dimensions.width / (nodesInLayer.length + 1);
      nodesInLayer.forEach((node, nodeIndex) => {
        const nx = node.x ?? nodeWidth * (nodeIndex + 1);
        const ny = node.y ?? layerHeight * (layerIndex + 1);
        const dist = Math.sqrt((x - nx) ** 2 + (y - ny) ** 2);
        if (dist < (node.size ?? 8) + 5) {
          foundNode = node;
        }
      });
    });

    if (foundNode) {
      handleNodeHover(foundNode.id);
      setTooltipPos({ x: e.clientX, y: e.clientY });
    } else {
      handleNodeHover(null);
    }
  };

  const handleClick = () => {
    if (hoveredNode) {
      handleNodeClick(hoveredNode.id);
    } else {
      handleBackgroundClick();
    }
  };

  return (
    <div ref={containerRef} className="relative w-full h-full">
      <canvas ref={canvasRef} className="w-full h-full cursor-pointer" onMouseMove={handleMouseMove} onClick={handleClick} />
      <NodeTooltip node={hoveredNode} position={tooltipPos} visible={hoveredNode !== null} />
      <Legend colorMode={colorMode} collapsed={false} />
      <MetricsDashboard snapshot={snapshot} collapsed={false} />
    </div>
  );
}
