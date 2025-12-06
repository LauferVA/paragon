/**
 * Color mappings for node types and statuses
 */

export const NODE_TYPE_COLORS: Record<string, string> = {
  REQ: '#ef4444',
  DESIGN: '#f97316',
  CODE: '#14b8a6',
  TEST: '#1d4ed8',
  DOC: '#3b82f6',
  BUG: '#dc2626',
  EPIC: '#8b5cf6',
  DEFAULT: '#6b7280',
};

export const NODE_STATUS_COLORS: Record<string, string> = {
  pending: '#f59e0b',
  active: '#3b82f6',
  completed: '#22c55e',
  failed: '#ef4444',
  blocked: '#6b7280',
  DEFAULT: '#6b7280',
};

export const EDGE_TYPE_COLORS: Record<string, string> = {
  implements: '#14b8a6',
  traces_to: '#ef4444',
  depends_on: '#6b7280',
  tests: '#1d4ed8',
  documents: '#3b82f6',
  blocks: '#dc2626',
  DEFAULT: '#6b7280',
};

export const TELEOLOGY_STATUS_COLORS: Record<string, string> = {
  valid: '#22c55e',
  orphan: '#f59e0b',
  blocked: '#ef4444',
  unknown: '#6b7280',
};

export function getNodeColor(type: string, colorMode: 'type' | 'status', status?: string): string {
  if (colorMode === 'status' && status) {
    return NODE_STATUS_COLORS[status] || NODE_STATUS_COLORS.DEFAULT;
  }
  return NODE_TYPE_COLORS[type] || NODE_TYPE_COLORS.DEFAULT;
}

export function getEdgeColor(type: string): string {
  return EDGE_TYPE_COLORS[type] || EDGE_TYPE_COLORS.DEFAULT;
}
