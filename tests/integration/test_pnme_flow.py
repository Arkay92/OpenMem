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

def test_multi_role_query(temp_pnme):
    # Store a complete fact
    temp_pnme.store("alice", "works_at", "wonderland")
    
    # Query with only subject (Stage 10 multi-role handling)
    res = temp_pnme.query(subject="alice")
    assert len(res) > 0
    symbols = res[0]["extracted_symbols"]
    assert symbols["relation"]["symbol"] == "works_at"
    assert symbols["object"]["symbol"] == "wonderland"

def test_absorb(temp_pnme):
    # Test regex-based absorption (Stage 11 composite)
    text = "Bob likes pizza. Alice is a coder."
    count = temp_pnme.absorb(text)
    assert count >= 2
    
    res = temp_pnme.query(subject="bob", relation="likes")
    assert res[0]["record"].object == "pizza"

def test_safety_integration(temp_pnme):
    temp_pnme.store("user", "key", "sk-123456789012345678901234567890")
    res = temp_pnme.query(subject="user")[0]
    assert "REDACTED" in res["record"].object
