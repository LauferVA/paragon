# Frontend Integration Guide: Graph Highlighting

Quick reference for integrating the dialogue-to-graph highlighting system in the frontend.

## API Endpoints

### 1. Highlight Nodes (Click Handler)

**Endpoint:** `POST /api/graph/highlight`

**Use Case:** User clicks a message or node in the UI

**Request:**
```typescript
interface HighlightRequest {
  highlight_type: "message" | "node" | "edge";
  source_id: string;
  highlight_mode: "exact" | "related" | "dependent";
}
```

**Response:**
```typescript
interface HighlightResponse {
  nodes_to_highlight: string[];
  edges_to_highlight: Array<{
    source: string;
    target: string;
  }>;
  context: string;
  highlight_mode: string;
  highlight_type: string;
}
```

**Example:**
```typescript
// User clicks a message
async function handleMessageClick(messageId: string) {
  const response = await fetch('/api/graph/highlight', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      highlight_type: 'message',
      source_id: messageId,
      highlight_mode: 'related'  // or 'exact', 'dependent'
    })
  });

  const data = await response.json();

  // Highlight nodes in graph visualization
  highlightNodes(data.nodes_to_highlight);
  highlightEdges(data.edges_to_highlight);

  // Show context tooltip
  showTooltip(data.context);
}
```

### 2. Get Node Messages (Hover Handler)

**Endpoint:** `GET /api/nodes/{node_id}/messages`

**Use Case:** User hovers over a node to see which dialogue turns mention it

**Response:**
```typescript
interface NodeMessagesResponse {
  node_id: string;
  messages: Array<{
    message_id: string;
    content: string;
    created_at: string;
    created_by: string;
    message_type: string;
  }>;
  count: number;
}
```

**Example:**
```typescript
// User hovers over a node
async function handleNodeHover(nodeId: string) {
  const response = await fetch(`/api/nodes/${nodeId}/messages`);
  const data = await response.json();

  if (data.count > 0) {
    // Show tooltip with message list
    showHoverTooltip({
      title: `Referenced in ${data.count} message(s)`,
      messages: data.messages.map(m => ({
        text: m.content,
        author: m.created_by,
        time: formatTime(m.created_at)
      }))
    });
  }
}
```

### 3. Get Reverse Connections (Detailed View)

**Endpoint:** `GET /api/nodes/{node_id}/reverse-connections`

**Use Case:** User clicks "Show Details" on a node

**Response:**
```typescript
interface ReverseConnectionsResponse {
  node_id: string;
  referenced_in_messages: Array<{
    message_id: string;
    content: string;
    created_at: string;
    created_by: string;
  }>;
  incoming_edges: Array<{
    source: string;
    target: string;
    type: string;
    weight: number;
    source_node_type: string;
  }>;
  outgoing_edges: Array<{
    source: string;
    target: string;
    type: string;
    weight: number;
    target_node_type: string;
  }>;
  definition_location: string | null;
  last_modified_by: string;
  last_modified_at: string;
  message_count: number;
}
```

**Example:**
```typescript
// User clicks "Details" button
async function showNodeDetails(nodeId: string) {
  const response = await fetch(`/api/nodes/${nodeId}/reverse-connections`);
  const data = await response.json();

  // Show detailed panel
  openDetailPanel({
    nodeId: data.node_id,
    location: data.definition_location,
    modifiedBy: data.last_modified_by,
    modifiedAt: data.last_modified_at,
    referencedIn: data.referenced_in_messages,
    incomingEdges: data.incoming_edges,
    outgoingEdges: data.outgoing_edges
  });
}
```

## WebSocket Integration

### Subscribe to Highlight Events

**Use Case:** Multi-tab synchronization - highlights in one tab appear in all tabs

**Message Format:**
```typescript
interface HighlightEvent {
  type: "node_highlight";
  data: {
    nodes_to_highlight: string[];
    edges_to_highlight: Array<{
      source: string;
      target: string;
    }>;
    reason: string;
    highlight_mode: string;
    timestamp: string;
  };
}
```

**Example:**
```typescript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/api/viz/ws');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  if (message.type === 'node_highlight') {
    const { nodes_to_highlight, edges_to_highlight, reason } = message.data;

    // Apply highlights
    highlightNodes(nodes_to_highlight);
    highlightEdges(edges_to_highlight);

    // Show notification
    showNotification(`Highlight update: ${reason}`);
  }
};
```

## Highlight Modes

### Exact Mode
- Highlights only the clicked node/message
- Use for: Precise selection, single-node focus

### Related Mode
- Highlights the node + immediate dependencies (DEPENDS_ON, IMPLEMENTS edges)
- Use for: Understanding direct dependencies, tracing implementation

### Dependent Mode
- Highlights the node + all descendant nodes
- Use for: Impact analysis, "what depends on this?"

## UI Patterns

### Pattern 1: Message Panel Click
```
User Action: Click message in dialogue panel
Backend Call: POST /api/graph/highlight (highlight_type="message")
UI Update: Highlight nodes in graph, show context tooltip
```

### Pattern 2: Node Hover
```
User Action: Hover over node in graph
Backend Call: GET /api/nodes/{id}/messages
UI Update: Show tooltip with message list
```

### Pattern 3: Node Click with Mode Selection
```
User Action: Click node + select mode from dropdown
Backend Call: POST /api/graph/highlight (highlight_type="node", mode=selected)
UI Update: Highlight related nodes, update mode indicator
```

### Pattern 4: Edge Click
```
User Action: Click edge in graph
Backend Call: POST /api/graph/highlight (highlight_type="edge", source_id="src:tgt")
UI Update: Highlight both endpoints + related nodes
```

## Error Handling

```typescript
async function safeHighlight(request: HighlightRequest) {
  try {
    const response = await fetch('/api/graph/highlight', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const error = await response.json();

      if (response.status === 404) {
        showError('Node or message not found');
      } else if (response.status === 400) {
        showError('Invalid highlight request: ' + error.error);
      } else {
        showError('Highlighting failed: ' + error.error);
      }

      return null;
    }

    return await response.json();
  } catch (err) {
    showError('Network error: ' + err.message);
    return null;
  }
}
```

## Performance Tips

1. **Debounce hover events**
   ```typescript
   const debouncedHover = debounce(handleNodeHover, 300);
   ```

2. **Cache highlight results**
   ```typescript
   const highlightCache = new Map<string, HighlightResponse>();

   async function getCachedHighlight(key: string, fetcher: () => Promise<HighlightResponse>) {
     if (highlightCache.has(key)) {
       return highlightCache.get(key);
     }
     const result = await fetcher();
     highlightCache.set(key, result);
     return result;
   }
   ```

3. **Use exact mode for large graphs**
   - Start with "exact" mode
   - Let user expand to "related" or "dependent" if needed

4. **Lazy load details**
   - Call `/messages` endpoint only on hover
   - Call `/reverse-connections` only when opening detail panel

## Complete Example: React Component

```typescript
import { useState, useCallback } from 'react';

function GraphVisualization() {
  const [highlightedNodes, setHighlightedNodes] = useState<string[]>([]);
  const [highlightMode, setHighlightMode] = useState<'exact' | 'related' | 'dependent'>('related');

  const handleMessageClick = useCallback(async (messageId: string) => {
    const response = await fetch('/api/graph/highlight', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        highlight_type: 'message',
        source_id: messageId,
        highlight_mode: highlightMode
      })
    });

    if (response.ok) {
      const data = await response.json();
      setHighlightedNodes(data.nodes_to_highlight);
    }
  }, [highlightMode]);

  const handleNodeHover = useCallback(async (nodeId: string) => {
    const response = await fetch(`/api/nodes/${nodeId}/messages`);

    if (response.ok) {
      const data = await response.json();
      // Show tooltip with messages
      setTooltip({
        title: `${data.count} reference(s)`,
        messages: data.messages
      });
    }
  }, []);

  return (
    <div>
      <ModeSelector value={highlightMode} onChange={setHighlightMode} />
      <Graph
        highlightedNodes={highlightedNodes}
        onNodeHover={handleNodeHover}
      />
      <DialoguePanel onMessageClick={handleMessageClick} />
    </div>
  );
}
```

## Testing

### Test Scenarios

1. **Click message in dialogue**
   - Should highlight referenced nodes
   - Should show context tooltip
   - Should update across WebSocket tabs

2. **Hover over node**
   - Should show message list
   - Should handle nodes with no messages
   - Should debounce rapid hovers

3. **Click node with different modes**
   - Exact: Only the node
   - Related: Node + dependencies
   - Dependent: Node + dependents

4. **Error cases**
   - Nonexistent node/message
   - Network errors
   - Invalid mode selection

## Support

- **Documentation:** `/docs/dialogue_to_graph_highlighting.md`
- **API Tests:** `/tests/unit/api/test_graph_highlighting.py`
- **Integration Tests:** `/tests/integration/test_dialogue_to_graph_mapping.py`
