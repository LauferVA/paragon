import { VizNode } from '../types/graph.types';

export interface LayoutBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  width: number;
  height: number;
}

export function computeBounds(nodes: VizNode[]): LayoutBounds {
  if (nodes.length === 0) {
    return { minX: 0, maxX: 100, minY: 0, maxY: 100, width: 100, height: 100 };
  }

  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;

  for (const node of nodes) {
    const x = node.x ?? 0;
    const y = node.y ?? 0;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }

  return {
    minX,
    maxX,
    minY,
    maxY,
    width: maxX - minX || 100,
    height: maxY - minY || 100,
  };
}

export function getLayerY(layer: number, totalLayers: number, height: number): number {
  if (totalLayers <= 1) return height / 2;
  return (layer / (totalLayers - 1)) * height * 0.8 + height * 0.1;
}

export function distributeNodesInLayer(nodes: VizNode[], layerY: number, width: number): void {
  const count = nodes.length;
  if (count === 0) return;

  const spacing = width / (count + 1);
  nodes.forEach((node, i) => {
    node.x = spacing * (i + 1);
    node.y = layerY;
  });
}

export function hierarchicalLayout(nodes: VizNode[], width: number, height: number): void {
  const layers = new Map<number, VizNode[]>();

  for (const node of nodes) {
    const layer = node.layer ?? 0;
    if (!layers.has(layer)) {
      layers.set(layer, []);
    }
    layers.get(layer)!.push(node);
  }

  const totalLayers = layers.size;
  const sortedLayers = Array.from(layers.keys()).sort((a, b) => a - b);

  for (const layerNum of sortedLayers) {
    const layerNodes = layers.get(layerNum)!;
    const layerY = getLayerY(layerNum, totalLayers, height);
    distributeNodesInLayer(layerNodes, layerY, width);
  }
}
