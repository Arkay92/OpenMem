import json
import numpy as np
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

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
    vector: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate record integrity."""
        if not self.subject or not self.relation or not self.object:
            raise ValueError("Triples must have non-empty subject, relation, and object.")
        
        if self.confidence < 0 or self.confidence > 1:
            self.confidence = max(0.0, min(1.0, self.confidence))
            
        if self.strength < 0:
            self.strength = 0.0

    def to_dict(self):
        """Standard dictionary for engine logic."""
        d = self.__dict__.copy()
        return d

    def to_storage_dict(self):
        """Prepare record for SQLite storage."""
        d = self.to_dict()
        # Serialize tags
        d['tags'] = json.dumps(self.tags)
        # Serialize vector
        if self.vector is not None:
            d['vector'] = self.vector.tobytes()
        return d

    @classmethod
    def from_row(cls, row):
        """Create a record from a SQLite Row object."""
        # Convert row to dict
        d = dict(row)
        
        # Deserialize tags
        if 'tags' in d and d['tags']:
            try:
                d['tags'] = json.loads(d['tags'])
            except:
                d['tags'] = []
        
        # Deserialize vector
        if 'vector' in d and d['vector']:
            d['vector'] = np.frombuffer(d['vector'], dtype=np.int8)
            
        return cls(**d)
