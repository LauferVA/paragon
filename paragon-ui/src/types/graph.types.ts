/**
 * Graph Type Definitions for Paragon UI
 * Corresponds to backend models in core/graph_db.py
 */

export interface VizNode {
  id: string;
  type: string;
  status: string;
  label: string;
  color: string;
  size: number;
  x?: number;
  y?: number;
  created_by: string;
  created_at: string;
  teleology_status: string;
  layer: number;
  is_root: boolean;
  is_leaf: boolean;
}

export interface VizEdge {
  source: string;
  target: string;
  type: string;
  color: string;
  weight: number;
}

export interface GraphSnapshot {
  timestamp: string;
  node_count: number;
  edge_count: number;
  nodes: VizNode[];
  edges: VizEdge[];
  layer_count: number;
  has_cycle: boolean;
  root_count: number;
  leaf_count: number;
  version: string;
  label: string;
}

export interface GraphDelta {
  timestamp: string;
  sequence: number;
  nodes_added: VizNode[];
  nodes_updated: VizNode[];
  nodes_removed: string[];
  edges_added: VizEdge[];
  edges_removed: { source: string; target: string }[];
}

export interface GraphMetrics {
  node_count: number;
  edge_count: number;
  layer_count: number;
  root_count: number;
  leaf_count: number;
  has_cycle: boolean;
  density: number;
  avg_degree: number;
}

export interface LayerInfo {
  layer_num: number;
  node_count: number;
  color: string;
  label: string;
}

export type NodeType = 'REQ' | 'DESIGN' | 'CODE' | 'TEST' | 'DOC' | 'BUG' | 'EPIC';
export type NodeStatus = 'pending' | 'active' | 'completed' | 'failed' | 'blocked';
export type EdgeType = 'depends_on' | 'implements' | 'tests' | 'documents' | 'blocks';
export type TeleologyStatus = 'valid' | 'orphan' | 'blocked' | 'unknown';
