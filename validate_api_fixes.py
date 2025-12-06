#!/usr/bin/env python
"""Quick validation check for API fixes."""

print("=" * 70)
print("API FIXES VALIDATION CHECK")
print("=" * 70)

# Check 1: GraphDelta schema
print("\n1. Checking GraphDelta.edges_removed schema...")
from viz.core import GraphDelta
from datetime import datetime, timezone

delta = GraphDelta(
    timestamp=datetime.now(timezone.utc).isoformat(),
    sequence=1,
    edges_removed=[{"source": "a", "target": "b"}]
)
delta_dict = delta.to_dict()
assert isinstance(delta_dict["edges_removed"][0], dict), "FAILED: edges_removed not dict"
print("   ✓ edges_removed uses dict format (not tuples)")

# Check 2: VizNode positions
print("\n2. Checking VizNode position hints...")
from viz.core import VizNode
node = VizNode(id="test", type="CODE", status="PENDING", label="test",
               color="#000", x=100.0, y=200.0)
assert node.x == 100.0 and node.y == 200.0, "FAILED: positions not set"
print("   ✓ VizNode supports x, y position hints")

# Check 3: Snapshot creation
print("\n3. Checking create_snapshot_from_db assigns positions...")
from viz.core import create_snapshot_from_db
from core.graph_db import ParagonDB
from core.schemas import NodeData
from core.ontology import NodeType

db = ParagonDB()
node = NodeData.create(type=NodeType.CODE.value, content="test", created_by="test")
db.add_node(node)
snapshot = create_snapshot_from_db(db)
assert snapshot.nodes[0].x is not None, "FAILED: x position not assigned"
assert snapshot.nodes[0].y is not None, "FAILED: y position not assigned"
print("   ✓ create_snapshot_from_db assigns position hints")

# Check 4: API routes
print("\n4. Checking API routes...")
from api.routes import create_app
app = create_app()
route_paths = [r.path for r in app.routes]
assert "/api/dialector/questions" in route_paths, "FAILED: dialector/questions missing"
assert "/api/dialector/answer" in route_paths, "FAILED: dialector/answer missing"
assert "/api/orchestrator/state" in route_paths, "FAILED: orchestrator/state missing"
print("   ✓ Dialectic endpoints registered")
print(f"   ✓ Total routes: {len(app.routes)}")

# Check 5: Sequence counter
print("\n5. Checking sequence counter...")
from api.routes import _next_sequence
seq1 = _next_sequence()
seq2 = _next_sequence()
assert seq2 > seq1, "FAILED: sequence not incrementing"
print("   ✓ Sequence counter increments")

# Check 6: Broadcast delta imports
print("\n6. Checking broadcast_delta and imports...")
from api.routes import broadcast_delta
print("   ✓ broadcast_delta function exists")

print("\n" + "=" * 70)
print("ALL VALIDATION CHECKS PASSED ✓")
print("=" * 70)
print("\nBackend is ready for frontend development!")
print("\nNext steps:")
print("1. Run: python validate_api_fixes.py")
print("2. Run: pytest tests/unit/api/test_routes.py")
print("3. Start server: granian --interface asgi api.routes:app")
print()
