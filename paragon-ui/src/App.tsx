import React, { useEffect } from 'react';
import { Layout } from './components/Layout';
import { GraphViewer } from './components/GraphViewer';
import { DialecticChat } from './components/DialecticChat';
import { useGraphWebSocket } from './hooks/useGraphWebSocket';
import { useGraphSnapshot } from './hooks/useGraphSnapshot';
import { useUIStore } from './stores/uiStore';

function App() {
  const { theme } = useUIStore();
  const { loading, error } = useGraphSnapshot();

  // Initialize WebSocket connection
  useGraphWebSocket();

  // Apply theme to document
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  if (error) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-900 text-white">
        <div className="text-center">
          <h1 className="text-xl font-bold mb-2">Connection Error</h1>
          <p className="text-gray-400">{error.message}</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-blue-600 rounded hover:bg-blue-500"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-900 text-white">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <p className="text-gray-400">Loading graph...</p>
        </div>
      </div>
    );
  }

  return (
    <Layout
      graphPanel={<GraphViewer />}
      dialecticPanel={<DialecticChat />}
    />
  );
}

export default App;
