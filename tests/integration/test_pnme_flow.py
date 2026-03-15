import os
import pytest
from pnme.api import PNME

@pytest.fixture
def temp_pnme():
    db_path = "test_integration.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    pnme = PNME(db_path=db_path)
    yield pnme
    if os.path.exists(db_path):
        os.remove(db_path)

def test_full_write_query_clear(temp_pnme):
    # 1. Write
    mid = temp_pnme.store("clippy", "is", "a paperclip")
    assert mid is not None
    
    # 2. Query Symbolic
    res = temp_pnme.query(subject="clippy", relation="is")
    assert len(res) > 0
    assert res[0]["record"].object == "a paperclip"
    
    # 3. Query HDC Context
    ctx = temp_pnme.get_context(["paperclip"])
    assert len(ctx) > 0

def test_hydration(temp_pnme):
    temp_pnme.store("clippy", "is", "a paperclip")
    prompt = "Tell me about clippy"
    hydrated = temp_pnme.hydrate(prompt)
    assert "a paperclip" in hydrated

def test_safety_integration(temp_pnme):
    temp_pnme.store("user", "key", "sk-123456789012345678901234567890")
    res = temp_pnme.query(subject="user")[0]
    assert "REDACTED" in res["record"].object
