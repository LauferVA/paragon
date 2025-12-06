import React from 'react';
import { useUIStore } from '../stores/uiStore';
import { NODE_TYPE_COLORS, NODE_STATUS_COLORS, EDGE_TYPE_COLORS } from '../utils/colorMaps';

interface LegendProps {
  colorMode: 'type' | 'status';
  collapsed?: boolean;
  onToggle?: () => void;
}

export function Legend({ colorMode, collapsed, onToggle }: LegendProps) {
  const { theme, legendCollapsed, toggleLegend } = useUIStore();
  const isCollapsed = collapsed ?? legendCollapsed;
  const handleToggle = onToggle ?? toggleLegend;

  const nodeColors = colorMode === 'status' ? NODE_STATUS_COLORS : NODE_TYPE_COLORS;
  const btnClass = theme === 'dark' ? 'bg-gray-800 text-gray-300 hover:bg-gray-700' : 'bg-gray-100 text-gray-700 hover:bg-gray-200';
  const boxClass = theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200';
  const labelClass = theme === 'dark' ? 'text-gray-400' : 'text-gray-500';
  const hoverClass = theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-100';

  if (isCollapsed) {
    return (
      <button onClick={handleToggle} className={'absolute bottom-2 left-2 px-2 py-1 rounded text-xs ' + btnClass}>
        Legend
      </button>
    );
  }

  return (
    <div className={'absolute bottom-2 left-2 p-3 rounded-lg shadow-lg border ' + boxClass}>
      <div className="flex items-center justify-between mb-2">
        <span className="font-bold text-xs">LEGEND</span>
        <button onClick={handleToggle} className={'text-xs px-1 rounded ' + hoverClass}>-</button>
      </div>

      <div className="mb-3">
        <div className={'text-xs font-medium mb-1 ' + labelClass}>
          {colorMode === 'status' ? 'NODE STATUS' : 'NODE TYPES'}
        </div>
        <div className="space-y-1">
          {Object.entries(nodeColors).filter(([k]) => k !== 'DEFAULT').map(([key, color]) => (
            <div key={key} className="flex items-center gap-2 text-xs">
              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
              <span className="capitalize">{key.toLowerCase().replace('_', ' ')}</span>
            </div>
          ))}
        </div>
      </div>

      <div>
        <div className={'text-xs font-medium mb-1 ' + labelClass}>EDGE TYPES</div>
        <div className="space-y-1">
          {Object.entries(EDGE_TYPE_COLORS).filter(([k]) => k !== 'DEFAULT').map(([key, color]) => (
            <div key={key} className="flex items-center gap-2 text-xs">
              <span className="w-4 h-0.5" style={{ backgroundColor: color }} />
              <span className="capitalize">{key.toLowerCase().replace('_', ' ')}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
