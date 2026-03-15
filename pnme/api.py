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
    def __init__(self, db_path: str = "pnme_memory.db", dim: int = 10000, anthropic_key: Optional[str] = None):
        self.engine = PNMEEngine(db_path, dim)
        self.hydrator = ContextHydrator(self.engine, max_tokens=800)
        self.extractor = MemoryExtractor(anthropic_key=anthropic_key)
        self.calibrator = Calibrator(self.engine)

    def store(self, subject: str, relation: str, obj: str, **kwargs) -> Dict[str, Any]:
        """Store a semantic triple. Returns a structured result with memory_id."""
        memory_id = self.engine.write(subject, relation, obj, **kwargs)
        return {"memory_id": memory_id, "status": "stored"}

    def absorb(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Extract facts from text and store them.
        Returns a summary result.
        """
        triples = self.extractor.extract_triples(text)
        ids = []
        for s, r, o in triples:
            ids.append(self.engine.write(s, r, o, **kwargs))
        return {"count": len(triples), "memory_ids": ids}

    def query(self, 
              query_text: Optional[str] = None,
              subject: Optional[str] = None, 
              relation: Optional[str] = None, 
              obj: Optional[str] = None,
              top_k: int = 5,
              include_vectors: bool = False) -> List[Dict[str, Any]]:
        """Query memory using natural language or partial triples. Returns slim results."""
        results = self.engine.query(query_text=query_text, subject=subject, relation=relation, obj=obj, top_k=top_k)
        
        # Slim down results for production API
        slim_results = []
        for r in results:
            record = r["record"]
            data = {
                "memory_id": record.memory_id,
                "subject": record.subject,
                "relation": record.relation,
                "object": record.object,
                "score": round(r["score"], 4),
                "explanation": r.get("explanation", {}),
                "metadata": {
                    "source": record.source,
                    "tags": record.tags,
                    "timestamp": record.timestamp_created,
                    "strength": record.strength
                }
            }
            if "extracted_symbols" in r:
                data["extracted_symbols"] = r["extracted_symbols"]
                
            if include_vectors:
                data["vector"] = record.vector.tolist() if hasattr(record.vector, "tolist") else record.vector
            
            slim_results.append(data)
        return slim_results

    def hydrate(self, prompt: str, top_k: int = 5) -> str:
        """Inject relevant long-term context into a prompt."""
        return self.hydrator.hydrate_context(prompt, top_k=top_k)

    def get_context(self, keywords: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for a set of keywords. Returns slim results."""
        results = self.engine.get_context(keywords, top_k=top_k)
        
        slim_results = []
        for r in results:
            record = r["record"]
            data = {
                "memory_id": record.memory_id,
                "subject": record.subject,
                "relation": record.relation,
                "object": record.object,
                "score": round(r["score"], 4),
                "metadata": {
                    "source": record.source,
                    "tags": record.tags,
                    "timestamp": record.timestamp_created,
                    "strength": record.strength
                }
            }
            slim_results.append(data)
        return slim_results

    def retrieve_context(self, keywords: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Alias for get_context."""
        return self.get_context(keywords, top_k=top_k)

    def export_data(self, path: str):
        """Export all memory data to JSONL."""
        self.engine.export_memory(path)

    def import_data(self, path: str):
        """Import memory data from JSONL."""
        self.engine.import_memory(path)
