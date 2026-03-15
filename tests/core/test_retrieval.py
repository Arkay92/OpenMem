import pytest
import numpy as np
from pnme.core.retrieval import RetrievalPipeline
from pnme.core.ranker import Ranker
from pnme.hdc.encoder import HDCEncoder
from pnme.storage.sqlite_store import SQLiteStore
from pnme.core.schema import MemoryRecord
from pnme.core.lifecycle import MemoryLifecycle
import os

@pytest.fixture
def retrieval_setup():
    db_path = "test_retrieval.db"
    if os.path.exists(db_path): os.remove(db_path)
    store = SQLiteStore(db_path)
    encoder = HDCEncoder(dim=1024) # Small dim for faster tests
    ranker = Ranker()
    lifecycle = MemoryLifecycle(store)
    pipeline = RetrievalPipeline(store, encoder, ranker)
    yield pipeline, store, encoder, lifecycle
    if os.path.exists(db_path): os.remove(db_path)

def test_execute_query_symbolic(retrieval_setup):
    pipeline, store, encoder, lifecycle = retrieval_setup # Added lifecycle to unpack
    
    record = MemoryRecord(
        subject="apple", relation="is", object="red",
        vector=encoder.encode_triple("apple", "is", "red"),
        source="test",
        strength=1.2 # Within valid range [0, 1.5]
    )
    store.store_memory_record(record)

    # Force a decay by running the logic
    lifecycle.apply_decay()
    
    updated = store.get_all_records()[0]
    assert updated.strength < 1.2

    # Test reinforcement
    initial_strength = record.strength
    lifecycle.reinforce(record)
    
    # Check updated record from store
    updated = store.get_all_records()[0]
    assert updated.strength > initial_strength
    assert updated.reinforcement_count == 1
    
    # Query
    results = pipeline.execute_query(subject="apple")
    assert len(results) > 0
    assert results[0]["record"].subject == "apple"
    assert "relation" in results[0]["extracted_symbols"]
    assert results[0]["extracted_symbols"]["relation"]["symbol"] == "is"

def test_associative_recall(retrieval_setup):
    pipeline, store, encoder, lifecycle = retrieval_setup # Added lifecycle to unpack
    
    store.store_memory_record(MemoryRecord(
        subject="banana", relation="is", object="yellow",
        vector=encoder.encode_triple("banana", "is", "yellow")
    ))
    
    # Associative recall via keywords
    results = pipeline.get_associative_context(["yellow"])
    assert len(results) > 0
    assert results[0]["record"].subject == "banana"
