"""
Research Feedback API Endpoints

New endpoints for research task feedback system.
These should be integrated into api/routes.py
"""
from starlette.requests import Request
from starlette.responses import JSONResponse
from datetime import datetime, timezone
import logging

from core.graph_db import ParagonDB
from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, NodeStatus, EdgeType
from viz.core import GraphDelta, VizNode, VizEdge
from agents.tools import get_db

# Event bus for real-time graph change notifications
try:
    from infrastructure.event_bus import get_event_bus, GraphEvent, EventType
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False


logger = logging.getLogger("paragon.api.research_feedback")


async def get_research_task(request: Request) -> JSONResponse:
    """
    Get a single research task with full metadata.

    Path params:
        research_task_id: The RESEARCH node ID

    Returns:
        {
            "node_id": "...",
            "req_node_id": "...",
            "iteration": 1,
            "query": "...",
            "total_findings": 5,
            "total_ambiguities": 2,
            "blocking_count": 0,
            "out_of_scope": [],
            "synthesis": "...",
            "findings": [...],
            "ambiguities": [...],
            "search_results": [...],
            "created_at": "...",
            "status": "...",
            "user_approval_required": bool,
            "user_approval_state": "pending|approved|denied|revision_requested",
            "user_feedback": "...",
            "user_feedback_timestamp": "...",
            "awaiting_user_action": bool,
            "hover_metadata": {
                "findings_tooltip": "...",
                "ambiguities_tooltip": "...",
                "synthesis_tooltip": "..."
            }
        }
    """
    from api.routes import error_response

    research_task_id = request.path_params["research_task_id"]
    db = get_db()

    try:
        # Get the research node
        node = db.get_node(research_task_id)

        if node.type != NodeType.RESEARCH.value:
            return error_response(f"Node {research_task_id} is not a RESEARCH node", status_code=400)

        # Find the REQ node this research is for
        req_node_id = ""
        all_edges = db.get_all_edges()
        for edge in all_edges:
            if (edge.type == EdgeType.RESEARCH_FOR.value and
                edge.source_id == node.id):
                req_node_id = edge.target_id
                break

        # Extract data from node
        data = node.data if node.data else {}

        # Build hover metadata for UI tooltips
        findings = data.get("findings", [])
        ambiguities = data.get("ambiguities", [])

        hover_metadata = {
            "findings_tooltip": f"{len(findings)} findings from research",
            "ambiguities_tooltip": f"{len(ambiguities)} ambiguities detected",
            "synthesis_tooltip": "Research synthesis and conclusions"
        }

        research_item = {
            "node_id": node.id,
            "req_node_id": req_node_id,
            "iteration": data.get("iteration", 0),
            "query": data.get("query", ""),
            "total_findings": data.get("total_findings", len(findings)),
            "total_ambiguities": data.get("total_ambiguities", len(ambiguities)),
            "blocking_count": data.get("blocking_count", 0),
            "out_of_scope": data.get("out_of_scope", []),
            "synthesis": node.content,
            "findings": findings,
            "ambiguities": ambiguities,
            "search_results": data.get("search_results", []),
            "created_at": node.created_at,
            "status": node.status,
            "user_approval_required": data.get("user_approval_required", False),
            "user_approval_state": data.get("user_approval_state", "pending"),
            "user_feedback": data.get("user_feedback"),
            "user_feedback_timestamp": data.get("user_feedback_timestamp"),
            "user_feedback_node_id": data.get("user_feedback_node_id"),
            "awaiting_user_action": data.get("awaiting_user_action", False),
            "hover_metadata": hover_metadata,
        }

        return JSONResponse(research_item)

    except Exception as e:
        return error_response(f"Failed to retrieve research task: {e}", status_code=500)


async def submit_research_feedback(request: Request, _ws_connections, _next_sequence, broadcast_delta) -> JSONResponse:
    """
    Submit user feedback on a research task.

    Path params:
        research_task_id: The RESEARCH node ID

    Body:
        {
            "feedback": "User's feedback text",
            "metadata": {
                "rating": 1-5,
                "issues": ["issue1", "issue2"],
                ...
            }
        }

    Creates:
        - MESSAGE/FEEDBACK node with feedback content
        - HAS_FEEDBACK edge from RESEARCH -> FEEDBACK
        - Updates research node with feedback reference
        - Publishes RESEARCH_FEEDBACK_RECEIVED event
        - Broadcasts to WebSocket clients

    Returns:
        {
            "success": true,
            "feedback_node_id": "...",
            "research_node_id": "...",
            "message": "Feedback submitted successfully"
        }
    """
    from api.routes import error_response

    research_task_id = request.path_params["research_task_id"]
    db = get_db()

    try:
        # Parse request body
        body = await request.json()
        feedback_text = body.get("feedback", "")
        metadata = body.get("metadata", {})

        if not feedback_text:
            return error_response("Feedback text is required", status_code=400)

        # Get the research node
        research_node = db.get_node(research_task_id)
        if research_node.type != NodeType.RESEARCH.value:
            return error_response(f"Node {research_task_id} is not a RESEARCH node", status_code=400)

        # Create FEEDBACK node (using MESSAGE type)
        import time
        feedback_node = NodeData.create(
            type=NodeType.MESSAGE.value,
            content=feedback_text,
            status=NodeStatus.VERIFIED.value,
            data={
                "message_type": "research_feedback",
                "source_agent": "user",
                "target_agent": "researcher",
                "research_node_id": research_task_id,
                "feedback_metadata": metadata,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            created_by="user",
        )
        db.add_node(feedback_node)

        # Create HAS_FEEDBACK edge
        feedback_edge = EdgeData.create(
            source_id=research_task_id,
            target_id=feedback_node.id,
            edge_type=EdgeType.HAS_FEEDBACK.value,
            created_by="user",
        )
        db.add_edge(feedback_edge)

        # Update research node with feedback reference
        research_data = research_node.data or {}
        research_data["user_feedback"] = feedback_text
        research_data["user_feedback_timestamp"] = datetime.now(timezone.utc).isoformat()
        research_data["user_feedback_node_id"] = feedback_node.id
        research_data["awaiting_user_action"] = False  # Feedback received, not awaiting anymore

        updated_research_node = NodeData.create(
            id=research_node.id,
            type=research_node.type,
            content=research_node.content,
            status=research_node.status,
            data=research_data,
            created_by=research_node.created_by,
            created_at=research_node.created_at,
        )
        db.update_node(research_task_id, updated_research_node)

        # Publish event
        if EVENT_BUS_AVAILABLE:
            event_bus = get_event_bus()
            event_bus.publish(GraphEvent(
                type=EventType.RESEARCH_FEEDBACK_RECEIVED,
                payload={
                    "research_node_id": research_task_id,
                    "feedback_node_id": feedback_node.id,
                    "feedback_text": feedback_text,
                    "metadata": metadata,
                },
                timestamp=time.time(),
                source="api"
            ))

        # Broadcast to WebSocket clients
        if _ws_connections:
            delta = GraphDelta(
                timestamp=datetime.now(timezone.utc).isoformat(),
                sequence=_next_sequence(),
                nodes_added=[VizNode.from_node_data(feedback_node)],
                edges_added=[VizEdge.from_edge_data(feedback_edge)],
                nodes_updated=[VizNode.from_node_data(updated_research_node)],
            )
            await broadcast_delta(delta)

        return JSONResponse({
            "success": True,
            "feedback_node_id": feedback_node.id,
            "research_node_id": research_task_id,
            "message": "Feedback submitted successfully"
        })

    except Exception as e:
        logger.error(f"Failed to submit research feedback: {e}", exc_info=True)
        return error_response(f"Failed to submit feedback: {e}", status_code=500)


async def submit_research_response(request: Request, _ws_connections, _next_sequence, broadcast_delta) -> JSONResponse:
    """
    Submit structured response to research task (approve/revise/clarify).

    Path params:
        research_task_id: The RESEARCH node ID

    Body:
        {
            "action": "approve" | "revise" | "clarify",
            "message": "Optional message to orchestrator",
            "context": {
                "revision_notes": "...",
                "clarification_needed": ["item1", "item2"],
                ...
            }
        }

    Actions:
        - approve: Mark research as approved, trigger next orchestrator phase
        - revise: Request revision with specific notes
        - clarify: Request clarification on specific items

    Returns:
        {
            "success": true,
            "action": "approve",
            "message_id": "...",
            "phase_transition": "RESEARCH -> PLAN" | null
        }
    """
    from api.routes import error_response

    research_task_id = request.path_params["research_task_id"]
    db = get_db()

    try:
        # Parse request body
        body = await request.json()
        action = body.get("action", "")
        message = body.get("message", "")
        context = body.get("context", {})

        if action not in ["approve", "revise", "clarify"]:
            return error_response("Action must be 'approve', 'revise', or 'clarify'", status_code=400)

        # Get the research node
        research_node = db.get_node(research_task_id)
        if research_node.type != NodeType.RESEARCH.value:
            return error_response(f"Node {research_task_id} is not a RESEARCH node", status_code=400)

        # Send agent message
        from agents.agent_messages import send_agent_message, AgentMessage

        agent_msg = AgentMessage(
            source_agent="user",
            target_agent="orchestrator",
            message_type=f"research_{action}",
            content=message or f"User {action}d research task",
            context={
                "research_node_id": research_task_id,
                "action": action,
                **context,
            },
            priority=1,  # High priority for user responses
        )
        message_id = send_agent_message(db, agent_msg)

        # Update research node state based on action
        research_data = research_node.data or {}

        if action == "approve":
            research_data["user_approval_state"] = "approved"
            research_data["user_approval_required"] = False
            research_data["awaiting_user_action"] = False
            # Update status to VERIFIED to signal completion
            new_status = NodeStatus.VERIFIED.value
            phase_transition = "RESEARCH -> PLAN"
        elif action == "revise":
            research_data["user_approval_state"] = "revision_requested"
            research_data["awaiting_user_action"] = True
            new_status = NodeStatus.PENDING.value
            phase_transition = None
        else:  # clarify
            research_data["user_approval_state"] = "denied"
            research_data["awaiting_user_action"] = True
            new_status = NodeStatus.PENDING.value
            phase_transition = None

        updated_research_node = NodeData.create(
            id=research_node.id,
            type=research_node.type,
            content=research_node.content,
            status=new_status,
            data=research_data,
            created_by=research_node.created_by,
            created_at=research_node.created_at,
        )
        db.update_node(research_task_id, updated_research_node)

        # Publish event if approved
        if action == "approve" and EVENT_BUS_AVAILABLE:
            import time
            event_bus = get_event_bus()
            event_bus.publish(GraphEvent(
                type=EventType.RESEARCH_TASK_COMPLETED,
                payload={
                    "research_node_id": research_task_id,
                    "approved": True,
                    "message_id": message_id,
                },
                timestamp=time.time(),
                source="api"
            ))

        # Broadcast to WebSocket clients
        if _ws_connections:
            delta = GraphDelta(
                timestamp=datetime.now(timezone.utc).isoformat(),
                sequence=_next_sequence(),
                nodes_updated=[VizNode.from_node_data(updated_research_node)],
            )
            await broadcast_delta(delta)

        return JSONResponse({
            "success": True,
            "action": action,
            "message_id": message_id,
            "phase_transition": phase_transition,
        })

    except Exception as e:
        logger.error(f"Failed to submit research response: {e}", exc_info=True)
        return error_response(f"Failed to submit response: {e}", status_code=500)


async def get_active_research_tasks(request: Request) -> JSONResponse:
    """
    Get all research tasks awaiting user feedback.

    Query params:
        limit: Max results (default 50)

    Returns:
        {
            "tasks": [
                {
                    "node_id": "...",
                    "req_node_id": "...",
                    "query": "...",
                    "synthesis": "...",
                    "total_findings": 5,
                    "total_ambiguities": 2,
                    "awaiting_user_action": true,
                    "user_approval_state": "pending",
                    "created_at": "...",
                    "action_hint": "Review research findings and approve or request revision"
                }
            ],
            "count": 3
        }
    """
    from api.routes import error_response

    db = get_db()
    limit = int(request.query_params.get("limit", 50))

    try:
        # Get all RESEARCH nodes
        all_nodes = db.get_all_nodes()
        research_nodes = [n for n in all_nodes if n.type == NodeType.RESEARCH.value]

        # Get all edges to find REQ relationships
        all_edges = db.get_all_edges()

        # Filter for active tasks (pending or awaiting user action)
        active_tasks = []
        for node in research_nodes:
            data = node.data or {}

            # Check if awaiting user action or status is PENDING
            if (data.get("awaiting_user_action", False) or
                node.status == NodeStatus.PENDING.value):

                # Find the REQ node this research is for
                req_node_id = ""
                for edge in all_edges:
                    if (edge.type == EdgeType.RESEARCH_FOR.value and
                        edge.source_id == node.id):
                        req_node_id = edge.target_id
                        break

                # Determine action hint based on state
                approval_state = data.get("user_approval_state", "pending")
                if approval_state == "revision_requested":
                    action_hint = "Revision requested - awaiting updated research"
                elif approval_state == "denied":
                    action_hint = "Clarification needed - awaiting response"
                else:
                    action_hint = "Review research findings and approve or request revision"

                task_item = {
                    "node_id": node.id,
                    "req_node_id": req_node_id,
                    "query": data.get("query", ""),
                    "synthesis": node.content[:200] + "..." if len(node.content) > 200 else node.content,
                    "total_findings": data.get("total_findings", 0),
                    "total_ambiguities": data.get("total_ambiguities", 0),
                    "awaiting_user_action": data.get("awaiting_user_action", False),
                    "user_approval_state": approval_state,
                    "created_at": node.created_at,
                    "action_hint": action_hint,
                }
                active_tasks.append(task_item)

        # Sort by created_at descending (newest first)
        active_tasks.sort(key=lambda x: x["created_at"], reverse=True)

        # Apply limit
        active_tasks = active_tasks[:limit]

        return JSONResponse({
            "tasks": active_tasks,
            "count": len(active_tasks)
        })

    except Exception as e:
        logger.error(f"Failed to get active research tasks: {e}", exc_info=True)
        return error_response(f"Failed to get active tasks: {e}", status_code=500)
