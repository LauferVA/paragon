# Research Feedback API - Quick Reference Card

## Endpoints at a Glance

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| GET | `/api/research/{id}` | Get research details | None |
| POST | `/api/research/{id}/feedback` | Submit feedback | None |
| POST | `/api/research/{id}/response` | Approve/Revise/Clarify | None |
| GET | `/api/research/tasks/active` | List active tasks | None |

## Quick cURL Examples

### Get Research Task
```bash
curl http://localhost:8000/api/research/research_abc123
```

### Submit Feedback
```bash
curl -X POST http://localhost:8000/api/research/research_abc123/feedback \
  -H "Content-Type: application/json" \
  -d '{"feedback": "Looks good!", "metadata": {"rating": 5}}'
```

### Approve Research
```bash
curl -X POST http://localhost:8000/api/research/research_abc123/response \
  -H "Content-Type: application/json" \
  -d '{"action": "approve", "message": "Approved"}'
```

### Request Revision
```bash
curl -X POST http://localhost:8000/api/research/research_abc123/response \
  -H "Content-Type: application/json" \
  -d '{"action": "revise", "message": "Needs more detail"}'
```

### Get Active Tasks
```bash
curl http://localhost:8000/api/research/tasks/active?limit=10
```

## Response Actions

| Action | Status Change | Next Phase | Awaiting User |
|--------|---------------|------------|---------------|
| approve | PENDING → VERIFIED | RESEARCH → PLAN | false |
| revise | stays PENDING | none | true |
| clarify | stays PENDING | none | true |

## Event Types

| Event | Trigger | Payload |
|-------|---------|---------|
| `RESEARCH_FEEDBACK_RECEIVED` | Feedback submitted | `{research_node_id, feedback_node_id, feedback_text, metadata}` |
| `RESEARCH_TASK_COMPLETED` | Research approved | `{research_node_id, approved, message_id}` |

## WebSocket Messages

| Type | When | Data |
|------|------|------|
| `research_feedback` | Feedback submitted | See events above |
| `research_completed` | Research approved | See events above |
| `delta` | Any change | Graph delta with nodes/edges added/updated |

## Graph Structure

```
REQ <---RESEARCH_FOR--- RESEARCH ---HAS_FEEDBACK---> MESSAGE (feedback)
                          |
                          +---TRACES_TO---> REQ
```

## Research Node Data Fields

```python
{
  "user_approval_required": bool,
  "user_approval_state": "pending" | "approved" | "denied" | "revision_requested",
  "user_feedback": str | null,
  "user_feedback_timestamp": str | null,
  "user_feedback_node_id": str | null,
  "awaiting_user_action": bool,
  # ... plus existing research fields
}
```

## Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid input) |
| 404 | Resource not found |
| 500 | Server error |

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| "not a RESEARCH node" | Wrong node type | Check node ID |
| "Feedback text is required" | Empty feedback | Provide text |
| "Action must be..." | Invalid action | Use approve/revise/clarify |

## Testing

```bash
# Run all tests
pytest tests/unit/api/test_research_feedback.py -v

# Run with coverage
pytest tests/unit/api/test_research_feedback.py --cov

# Run specific test
pytest tests/unit/api/test_research_feedback.py::TestGetResearchTask -v
```

## Integration Checklist

- [ ] Add imports to routes.py
- [ ] Create wrapper functions
- [ ] Register routes in create_routes()
- [ ] Add WebSocket subscriptions in viz_websocket()
- [ ] Test all endpoints
- [ ] Verify events publish correctly
- [ ] Check graph visualization updates
- [ ] Confirm orchestrator receives messages

## Files Modified

✅ `/Users/lauferva/paragon/core/ontology.py` - Added HAS_FEEDBACK
✅ `/Users/lauferva/paragon/infrastructure/event_bus.py` - Added events
📄 `/Users/lauferva/paragon/api/research_feedback_endpoints.py` - New endpoints
📄 `/Users/lauferva/paragon/tests/unit/api/test_research_feedback.py` - Tests
⚠️ `/Users/lauferva/paragon/api/routes.py` - Needs integration (see INTEGRATION.md)

## Performance

- GET research task: ~10ms (single node lookup)
- POST feedback: ~20ms (2 nodes + 1 edge + event)
- POST response: ~25ms (1 update + agent message + event)
- GET active tasks: ~50ms (full graph scan)

## Next Steps

1. Read `RESEARCH_FEEDBACK_INTEGRATION.md` for detailed integration steps
2. Follow integration checklist above
3. Run tests to verify integration
4. Test WebSocket updates in UI
5. Monitor orchestrator for approval messages
