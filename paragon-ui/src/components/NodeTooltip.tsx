import React from 'react';
import { VizNode } from '../types/graph.types';
import { useUIStore } from '../stores/uiStore';
import { TELEOLOGY_STATUS_COLORS } from '../utils/colorMaps';

interface NodeTooltipProps {
  node: VizNode | null;
  position: { x: number; y: number };
  visible: boolean;
}

export function NodeTooltip({ node, position, visible }: NodeTooltipProps) {
  const { theme } = useUIStore();

  if (!visible || !node) return null;

  const teleologyColor = TELEOLOGY_STATUS_COLORS[node.teleology_status] || TELEOLOGY_STATUS_COLORS.unknown;
  const bgClass = theme === 'dark' ? 'bg-gray-800 border-gray-700 text-white' : 'bg-white border-gray-200 text-gray-900';
  const subClass = theme === 'dark' ? 'text-gray-400' : 'text-gray-500';
  const borderClass = theme === 'dark' ? 'border-gray-700' : 'border-gray-200';
  const hintClass = theme === 'dark' ? 'text-gray-500' : 'text-gray-400';

  return (
    <div
      className={'absolute z-50 p-3 rounded-lg shadow-lg border max-w-xs pointer-events-none ' + bgClass}
      style={{ left: position.x + 10, top: position.y + 10 }}
    >
      <div className="font-bold text-sm mb-1">{node.type}: {node.label}</div>
      <div className={'text-xs mb-2 ' + subClass}>ID: {node.id.slice(0, 12)}...</div>

      <div className={'border-t pt-2 mb-2 ' + borderClass}>
        <div className="flex items-center gap-2 text-xs mb-1">
          <span className={subClass}>Status:</span>
          <span className="font-medium capitalize">{node.status}</span>
        </div>
        <div className="flex items-center gap-2 text-xs mb-1">
          <span className={subClass}>Created:</span>
          <span>{new Date(node.created_at).toLocaleString()}</span>
        </div>
        <div className="flex items-center gap-2 text-xs mb-1">
          <span className={subClass}>Agent:</span>
          <span className="font-mono text-xs">{node.created_by}</span>
        </div>
      </div>

      <div className={'border-t pt-2 ' + borderClass}>
        <div className="flex items-center gap-2 text-xs">
          <span className={subClass}>Teleology:</span>
          <span className="font-medium capitalize" style={{ color: teleologyColor }}>{node.teleology_status}</span>
        </div>
        <div className="flex items-center gap-2 text-xs mt-1">
          <span className={subClass}>Layer:</span>
          <span>{node.layer}</span>
          {node.is_root && <span className="px-1 bg-blue-600 text-white rounded text-xs">Root</span>}
          {node.is_leaf && <span className="px-1 bg-green-600 text-white rounded text-xs">Leaf</span>}
        </div>
      </div>

      <div className={'text-xs mt-2 italic ' + hintClass}>Click for details</div>
    </div>
  );
}
