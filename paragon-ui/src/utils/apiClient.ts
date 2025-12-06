import { GraphSnapshot } from '../types/graph.types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function fetchSnapshot(colorMode: 'type' | 'status' = 'type'): Promise<GraphSnapshot> {
  const response = await fetch(`${API_BASE_URL}/api/viz/snapshot?color_mode=${colorMode}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch snapshot: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchStats(): Promise<{ node_count: number; edge_count: number; has_cycle: boolean }> {
  const response = await fetch(`${API_BASE_URL}/stats`);
  if (!response.ok) {
    throw new Error(`Failed to fetch stats: ${response.statusText}`);
  }
  return response.json();
}

export async function healthCheck(): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }
  return response.json();
}

export function getWebSocketUrl(endpoint: string): string {
  const wsBase = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
  return `${wsBase}${endpoint}`;
}
