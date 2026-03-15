from .core.engine import PNMEEngine
from .core.hydrator import ContextHydrator
from .core.extractor import MemoryExtractor
from .core.calibration import Calibrator

class PNME:
    """High-level API for Persistent Neuro-Symbolic Memory Engine."""
    def __init__(self, db_path="pnme_memory.db", dim=10000):
        self.engine = PNMEEngine(db_path, dim)
        self.hydrator = ContextHydrator(self)
        self.extractor = MemoryExtractor()
        self.calibrator = Calibrator(self.engine)

    def store(self, subject, relation, obj, **kwargs):
        """Store a fact or episodic memory with optional metadata."""
        return self.engine.write(subject, relation, obj, **kwargs)

    def query(self, subject=None, relation=None, obj=None):
        """Query memory using partial triples."""
        return self.engine.query(subject=subject, relation=relation, obj=obj)

    def retrieve_context(self, keywords):
        """Get relevant memories for a set of keywords."""
        if isinstance(keywords, str):
            keywords = keywords.split()
        return self.engine.get_context(keywords)

    def associate(self, query_vector):
        """Associate a raw vector with stored memories (advanced)."""
        records = self.engine.store.get_all_records()
        memories = [r.to_dict() for r in records]
        for i, m in enumerate(memories):
            m['vector'] = records[i].vector
        from .core.recall import associate_recall
        return associate_recall(query_vector, memories)
