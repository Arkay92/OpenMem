import pytest
from pnme.core.lifecycle import MemoryLifecycle
from pnme.storage.sqlite_store import SQLiteStore
from pnme.core.schema import MemoryRecord
import os
import time

@pytest.fixture
def lifecycle_setup():
    db_path = "test_lifecycle.db"
    if os.path.exists(db_path): os.remove(db_path)
    store = SQLiteStore(db_path)
    lifecycle = MemoryLifecycle(store)
    yield lifecycle, store
    if os.path.exists(db_path): os.remove(db_path)

def test_reinforcement(lifecycle_setup):
    lifecycle, store = lifecycle_setup
    
    record = MemoryRecord(subject="test", relation="has", object="strength")
    store.store_memory_record(record)
    
    initial_strength = record.strength
    lifecycle.reinforce(record)
    
    # Check updated record from store
    updated = store.get_all_records()[0]
    assert updated.strength > initial_strength
    assert updated.reinforcement_count == 1

def test_decay(lifecycle_setup):
    lifecycle, store = lifecycle_setup
    
    record = MemoryRecord(subject="old", relation="is", object="forgotten", strength=2.0)
    store.store_memory_record(record)
    
    # Manually tweak last_accessed to make it older
    # Decay factor is 0.1 per hour by default.
    # We    # Force a decay by running the logic
    lifecycle.apply_decay()
    
    updated = store.get_all_records()[0]
    assert updated.strength < 2.0
