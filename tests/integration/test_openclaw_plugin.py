import pytest
from pnme.integrations.openclaw_plugin import PNMEPlugin
import os

@pytest.fixture
def openclaw_setup():
    db_path = "test_openclaw.db"
    if os.path.exists(db_path): os.remove(db_path)
    plugin = PNMEPlugin(db_path)
    yield plugin
    if os.path.exists(db_path): os.remove(db_path)

def test_openclaw_skills(openclaw_setup):
    plugin = openclaw_setup
    skills = plugin.get_skills()
    
    assert "store_memory" in skills
    assert "retrieve_context" in skills # Added alias (Stage 13)
    
    skills["store_memory"]("charlie", "is", "a cat")
    res = skills["query_memory"](subject="charlie")
    assert len(res) > 0
    assert res[0]["object"] == "a cat"

def test_openclaw_context(openclaw_setup):
    plugin = openclaw_setup
    skills = plugin.get_skills()
    
    skills["store_memory"]("charlie", "is", "a cat")
    ctx = skills["retrieve_context"](["cat"])
    assert len(ctx) > 0
    assert "charlie" in ctx[0]["subject"]
