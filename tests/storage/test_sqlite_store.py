import pytest
import os
import json
import numpy as np
from pnme.storage.sqlite_store import SQLiteStore
from pnme.core.schema import MemoryRecord

@pytest.fixture
def temp_store(tmp_path):
    db_file = tmp_path / "test_pnme.db"
    return SQLiteStore(str(db_file))

def test_init_db_v2(temp_store):
    # Verify tables exist
    with temp_store._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row['name'] for row in cursor.fetchall()]
        assert "memory_events" in tables
        assert "access_log" in tables
        assert "tombstones" in tables
        assert "settings" in tables
        
        cursor.execute("SELECT value FROM meta WHERE key='version'")
        assert cursor.fetchone()['value'] == '2'

def test_log_access(temp_store):
    temp_store.log_access("mem123", query_type="test")
    with temp_store._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM access_log WHERE memory_id='mem123'")
        row = cursor.fetchone()
        assert row is not None
        assert row['query_type'] == "test"

def test_jsonl_export_import(temp_store, tmp_path):
    # Create a record
    rec = MemoryRecord(
        subject="test",
        relation="is",
        object="working",
        vector=np.array([1, -1, 1], dtype=np.int8)
    )
    temp_store.store_memory_record(rec)
    
    export_file = tmp_path / "export.jsonl"
    temp_store.export_jsonl(str(export_file))
    
    # Check file
    assert export_file.exists()
    with open(export_file, 'r') as f:
        line = f.readline()
        data = json.loads(line)
        assert data['subject'] == "test"
        assert isinstance(data['vector'], list)
    
    # Import into a new store
    new_db = tmp_path / "new_pnme.db"
    new_store = SQLiteStore(str(new_db))
    new_store.import_jsonl(str(export_file))
    
    records = new_store.get_all_records()
    assert len(records) == 1
    assert records[0].subject == "test"
    assert np.array_equal(records[0].vector, rec.vector)

def test_settings(temp_store):
    temp_store.set_setting("dim", "10000")
    assert temp_store.get_setting("dim") == "10000"
    assert temp_store.get_setting("nonexistent", "default") == "default"
