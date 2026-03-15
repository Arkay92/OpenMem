from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid

@dataclass
class MemoryRecord:
    subject: str
    relation: str
    object: str
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: str = "semantic"  # episodic, semantic, procedural, associative
    source: str = "unknown"
    session_id: str = "default"
    agent_id: str = "default"
    context: str = ""
    timestamp_created: str = field(default_factory=lambda: datetime.now().isoformat())
    timestamp_last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 1.0
    strength: float = 1.0
    reinforcement_count: int = 0
    decay_factor: float = 0.001
    provenance: str = ""
    tags: List[str] = field(default_factory=list)
    vector_encoding_version: str = "1.0"
    symbolic_version: str = "1.0"
    privacy_level: int = 0  # 0: public, 1: internal, 2: secret
    vector: Optional[Any] = None

    def to_dict(self):
        d = self.__dict__.copy()
        if self.vector is not None:
            # We don't usually want to serialize the raw vector in a standard dict/json
            # unless explicitly needed (e.g. for storage).
            # But for the engine logic, we keep it.
            pass
        return d

    @classmethod
    def from_row(cls, row, row_mapping):
        """Helper to create a record from a SQLite row."""
        # This will be used in the storage layer
        pass
