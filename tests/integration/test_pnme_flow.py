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
    res_store = temp_pnme.store("clippy", "is", "a paperclip")
    assert "memory_id" in res_store
    
    # 2. Query Symbolic
    res = temp_pnme.query(subject="clippy", relation="is")
    assert len(res) > 0
    assert res[0]["object"] == "a paperclip"
    
    # 3. Query HDC Context
    ctx = temp_pnme.get_context(["paperclip"])
    assert len(ctx) > 0

def test_hydration(temp_pnme):
    temp_pnme.store("clippy", "is", "a paperclip")
    prompt = "Tell me about clippy"
    hydrated = temp_pnme.hydrate(prompt)
    assert "a paperclip" in hydrated

def test_regex_repairs(temp_pnme):
    # Test hyphens and dots (repaired regex Stage)
    text = "DeepSeek-V3 is a model. Claude-3.5 is fast."
    res_absorb = temp_pnme.absorb(text)
    assert res_absorb["count"] >= 2
    
    res = temp_pnme.query(subject="deepseek-v3")
    assert len(res) > 0
    assert res[0]["object"] == "a model"

def test_multi_role_query(temp_pnme):
    # Store a complete fact
    temp_pnme.store("alice", "works_at", "wonderland")
    
    # Query with only subject (Stage 10 multi-role handling)
    res = temp_pnme.query(subject="alice")
    assert len(res) > 0
    symbols = res[0]["extracted_symbols"]
    assert symbols["relation"]["symbol"] == "works_at"
    assert symbols["object"]["symbol"] == "wonderland"
    
    # Query with only relation
    res2 = temp_pnme.query(relation="works_at")
    assert len(res2) > 0
    symbols2 = res2[0]["extracted_symbols"]
    assert symbols2["subject"]["symbol"] == "alice"
    assert symbols2["object"]["symbol"] == "wonderland"

def test_safety_integration(temp_pnme):
    temp_pnme.store("user", "key", "sk-123456789012345678901234567890")
    res = temp_pnme.query(subject="user")[0]
    assert "REDACTED" in res["object"]
