"""
Pytest configuration and shared fixtures for Paragon test suite.
"""
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test to ensure isolation."""
    from agents.tools import set_db
    from core.graph_db import ParagonDB

    # Fresh database for each test
    set_db(ParagonDB())

    yield

    # Cleanup after test
    set_db(None)


@pytest.fixture
def fresh_db():
    """Provide a fresh ParagonDB instance."""
    from core.graph_db import ParagonDB
    return ParagonDB()


@pytest.fixture
def db_with_sample_nodes(fresh_db):
    """Provide a database with sample nodes for testing."""
    from core.schemas import NodeData
    from core.ontology import NodeType

    req = NodeData.create(type=NodeType.REQ.value, content="Build a calculator")
    spec = NodeData.create(type=NodeType.SPEC.value, content="Add two numbers")
    code = NodeData.create(type=NodeType.CODE.value, content="def add(a, b): return a + b")

    fresh_db.add_node(req)
    fresh_db.add_node(spec)
    fresh_db.add_node(code)

    return fresh_db, {"req": req, "spec": spec, "code": code}


@pytest.fixture
def mock_llm_disabled(monkeypatch):
    """Disable LLM for tests that don't need it."""
    import agents.orchestrator as orch
    monkeypatch.setattr(orch, "LLM_AVAILABLE", False)
