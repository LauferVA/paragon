import { useEffect, useState } from 'react';
import { useGraphStore } from '../stores/graphStore';
import { fetchSnapshot } from '../utils/apiClient';

export function useGraphSnapshot() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const { setSnapshot, colorMode } = useGraphStore();

  useEffect(() => {
    let cancelled = false;

    async function loadSnapshot() {
      try {
        setLoading(true);
        setError(null);
        const snapshot = await fetchSnapshot(colorMode);
        if (!cancelled) {
          setSnapshot(snapshot);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err : new Error('Failed to fetch snapshot'));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadSnapshot();

    return () => {
      cancelled = true;
    };
  }, [colorMode, setSnapshot]);

  return { loading, error };
}
