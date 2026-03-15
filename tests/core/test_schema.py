import pytest
from pnme.core.schema import MemoryRecord
import numpy as np

def test_memory_record_serialization():
    v = np.array([1.0, -1.0, 1.0])
    record = MemoryRecord(
        subject="user",
        relation="likes",
        object="rust",
        vector=v,
        confidence=0.9
    )
    
    data = record.to_dict()
    assert data["subject"] == "user"
    assert data["relation"] == "likes"
    assert data["object"] == "rust"
    assert data["confidence"] == 0.9
    # Vector is included in the dict for internal engine use
    assert "vector" in data

def test_memory_record_from_dict():
    data = {
        "subject": "agent",
        "relation": "version",
        "object": "1.0",
        "confidence": 1.0,
        "memory_id": "test-uuid"
    }
    # Note: from_dict doesn't exist yet, we usually use kwargs in constructor 
    # but let's test if we can reconstruct it
    record = MemoryRecord(**data)
    assert record.subject == "agent"
    assert record.memory_id == "test-uuid"
