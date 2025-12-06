import React from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { useUIStore } from '../stores/uiStore';
import { useGraphStore } from '../stores/graphStore';

interface LayoutProps {
  graphPanel: React.ReactNode;
  dialecticPanel: React.ReactNode;
}

export function Layout({ graphPanel, dialecticPanel }: LayoutProps) {
  const { theme, setTheme, splitRatio, setSplitRatio } = useUIStore();
  const { isConnected } = useGraphStore();

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  const bgClass = theme === 'dark' ? 'bg-gray-900 text-white' : 'bg-white text-gray-900';
  const headerClass = theme === 'dark' ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-gray-50';
  const tagClass = theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200';
  const btnClass = theme === 'dark' ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-200 hover:bg-gray-300';
  const resizeClass = theme === 'dark' ? 'bg-gray-700 hover:bg-blue-500' : 'bg-gray-300 hover:bg-blue-400';

  return (
    <div className={'h-screen flex flex-col ' + bgClass}>
      <header className={'flex items-center justify-between px-4 py-2 border-b ' + headerClass}>
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-bold">Paragon</h1>
          <span className={'text-xs px-2 py-0.5 rounded ' + tagClass}>v1.0</span>
          <span className={'text-xs px-2 py-0.5 rounded ' + (isConnected ? 'bg-green-600 text-white' : 'bg-red-600 text-white')}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={toggleTheme} className={'px-3 py-1 text-sm rounded ' + btnClass}>
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
        </div>
      </header>

      <main className="flex-1 overflow-hidden">
        <PanelGroup direction="horizontal" onLayout={(sizes) => setSplitRatio(sizes[0] / 100)}>
          <Panel defaultSize={splitRatio * 100} minSize={30}>
            <div className="h-full overflow-hidden">{graphPanel}</div>
          </Panel>
          <PanelResizeHandle className={'w-1 cursor-col-resize ' + resizeClass} />
          <Panel minSize={20}>
            <div className="h-full overflow-hidden">{dialecticPanel}</div>
          </Panel>
        </PanelGroup>
      </main>
    </div>
  );
}
