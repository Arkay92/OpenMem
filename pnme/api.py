from typing import List, Dict, Any, Optional
from .core.engine import PNMEEngine
from .core.hydrator import ContextHydrator
from .core.extractor import MemoryExtractor
from .core.calibration import Calibrator

class PNME:
    """
    High-level API for the Persistent Neuro-Symbolic Memory Engine.
    Designed to be used as a memory coprocessor for LLM agents.
    """
    def __init__(self, db_path: str = "pnme_memory.db", dim: int = 10000):
        self.engine = PNMEEngine(db_path, dim)
        self.hydrator = ContextHydrator(self)
        self.extractor = MemoryExtractor()
        self.calibrator = Calibrator(self.engine)

    def store(self, subject: str, relation: str, obj: str, **kwargs) -> str:
        """Store a semantic triple with optional metadata."""
        return self.engine.write(subject, relation, obj, **kwargs)

    def absorb(self, text: str, **kwargs) -> int:
        """
        Extract facts from text and store them as memories.
        Returns the number of facts stored.
        """
        triples = self.extractor.extract_triples(text)
        for s, r, o in triples:
            self.store(s, r, o, **kwargs)
        return len(triples)

    def query(self, 
              subject: Optional[str] = None, 
              relation: Optional[str] = None, 
              obj: Optional[str] = None,
              top_k: int = 5) -> List[Dict[str, Any]]:
        """Query memory using partial triples with hybrid ranking."""
        return self.engine.query(subject=subject, relation=relation, obj=obj, top_k=top_k)

    def hydrate(self, prompt: str, top_k: int = 5) -> str:
        """Inject relevant long-term context into a prompt."""
        return self.hydrator.hydrate_context(prompt, top_k=top_k)

    def get_context(self, keywords: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for a set of keywords."""
        return self.engine.get_context(keywords, top_k=top_k)

    def export_data(self, path: str):
        """Export all memory data to JSONL."""
        self.engine.export_memory(path)

    def import_data(self, path: str):
        """Import memory data from JSONL."""
        self.engine.import_memory(path)
