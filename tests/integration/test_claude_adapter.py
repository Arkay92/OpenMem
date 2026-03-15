import pytest
import json
from pnme.integrations.claude_tools import ClaudeMemoryAdapter
from pnme.api import PNME
import os

@pytest.fixture
def claude_setup():
    db_path = "test_claude.db"
    if os.path.exists(db_path): os.remove(db_path)
    pnme = PNME(db_path)
    adapter = ClaudeMemoryAdapter(pnme)
    yield adapter, pnme
    if os.path.exists(db_path): os.remove(db_path)

def test_claude_store_and_query(claude_setup):
    adapter, _ = claude_setup
    
    # Simulate tool call: memory_store
    adapter.handle_tool_call("memory_store", {"subject": "user", "relation": "loves", "object": "coding"})
    
    # Simulate tool call: memory_query
    res_json = adapter.handle_tool_call("memory_query", {"subject": "user"})
    res = json.loads(res_json)
    assert len(res) > 0
    assert "coding" in res[0]["fact"]

def test_claude_absorb(claude_setup):
    adapter, _ = claude_setup
    
    adapter.handle_tool_call("memory_absorb", {"text": "Alice moved to London. Bob works at Google."})
    
    res_json = adapter.handle_tool_call("memory_query", {"subject": "alice"})
    res = json.loads(res_json)
    assert "london" in res[0]["fact"].lower()
