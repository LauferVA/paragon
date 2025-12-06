import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UIStore {
  // State
  theme: 'dark' | 'light';
  splitRatio: number;
  metricsCollapsed: boolean;
  legendCollapsed: boolean;

  // Actions
  setTheme: (theme: 'dark' | 'light') => void;
  setSplitRatio: (ratio: number) => void;
  toggleMetrics: () => void;
  toggleLegend: () => void;
}

export const useUIStore = create<UIStore>()(
  persist(
    (set) => ({
      theme: 'dark',
      splitRatio: 0.6,
      metricsCollapsed: false,
      legendCollapsed: false,

      setTheme: (theme) => set({ theme }),
      setSplitRatio: (ratio) => set({ splitRatio: ratio }),
      toggleMetrics: () => set((state) => ({ metricsCollapsed: !state.metricsCollapsed })),
      toggleLegend: () => set((state) => ({ legendCollapsed: !state.legendCollapsed })),
    }),
    {
      name: 'paragon-ui-settings',
    }
  )
);
