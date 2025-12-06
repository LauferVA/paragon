import React from 'react';
import { GraphSnapshot } from '../types/graph.types';
import { useUIStore } from '../stores/uiStore';

interface MetricsDashboardProps {
  snapshot: GraphSnapshot | null;
  collapsed?: boolean;
  onToggle?: () => void;
}

export function MetricsDashboard({ snapshot, collapsed, onToggle }: MetricsDashboardProps) {
  const { theme, metricsCollapsed, toggleMetrics } = useUIStore();
  const isCollapsed = collapsed ?? metricsCollapsed;
  const handleToggle = onToggle ?? toggleMetrics;

  const btnClass = theme === 'dark' ? 'bg-gray-800 text-gray-300 hover:bg-gray-700' : 'bg-gray-100 text-gray-700 hover:bg-gray-200';
  const boxClass = theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200';
  const labelClass = theme === 'dark' ? 'text-gray-400' : 'text-gray-500';
  const hoverClass = theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-100';

  if (isCollapsed) {
    return (
      <button onClick={handleToggle} className={'absolute top-2 right-2 px-2 py-1 rounded text-xs ' + btnClass}>
        Metrics
      </button>
    );
  }

  return (
    <div className={'absolute top-2 right-2 p-3 rounded-lg shadow-lg border ' + boxClass}>
      <div className="flex items-center justify-between mb-2">
        <span className="font-bold text-sm">Metrics</span>
        <button onClick={handleToggle} className={'text-xs px-1 rounded ' + hoverClass}>-</button>
      </div>

      {snapshot ? (
        <div className="space-y-1 text-xs">
          <div className="flex justify-between gap-4">
            <span className={labelClass}>Nodes:</span>
            <span className="font-mono">{snapshot.node_count.toLocaleString()}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className={labelClass}>Edges:</span>
            <span className="font-mono">{snapshot.edge_count.toLocaleString()}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className={labelClass}>Layers:</span>
            <span className="font-mono">{snapshot.layer_count}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className={labelClass}>Roots:</span>
            <span className="font-mono">{snapshot.root_count}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className={labelClass}>Leaves:</span>
            <span className="font-mono">{snapshot.leaf_count}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className={labelClass}>Cyclic:</span>
            <span className={'font-mono ' + (snapshot.has_cycle ? 'text-red-500' : 'text-green-500')}>
              {snapshot.has_cycle ? 'Yes' : 'No'}
            </span>
          </div>
        </div>
      ) : (
        <div className={'text-xs ' + labelClass}>Loading...</div>
      )}
    </div>
  );
}
