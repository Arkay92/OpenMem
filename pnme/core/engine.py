from ..hdc.encoder import HDCEncoder
from ..storage.sqlite_store import SQLiteStore
from .schema import MemoryRecord
from .ranker import Ranker
from .safety import SafetyFilter
from .lifecycle import MemoryLifecycle
from .retrieval import RetrievalPipeline

import numpy as np
from datetime import datetime

class PNMEEngine:
    def __init__(self, db_path="pnme_memory.db", dim=10000):
        self.store = SQLiteStore(db_path)
        self.encoder = HDCEncoder(dim)
        self.ranker = Ranker()
        self.safety = SafetyFilter()
        
        # New Components
        self.lifecycle = MemoryLifecycle(self.store)
        self.retrieval = RetrievalPipeline(self.store, self.encoder, self.ranker)
        
        self._load_base_vectors()

    def _load_base_vectors(self):
        """Restore symbol map from storage."""
        symbols = self.store.load_symbols()
        self.encoder.symbol_map.update(symbols)

    def write(self, subject, relation, obj, **kwargs):
        """Store a semantic memory with rich metadata and safety scrubbing."""
        # Safety Scrubbing
        context = kwargs.get("context", "")
        scrubbed = self.safety.scrub_record(subject, relation, obj, context)
        
        subject = scrubbed["subject"]
        relation = scrubbed["relation"]
        obj = scrubbed["object"]
        if "context" in kwargs:
            kwargs["context"] = scrubbed["context"]

        # Encode
        vector = self.encoder.encode_triple(subject, relation, obj)
        
        # Create record
        record = MemoryRecord(
            subject=subject,
            relation=relation,
            object=obj,
            vector=vector,
            **kwargs
        )
        
        # Save memory
        self.store.store_memory_record(record)
        
        # Ensure all involved symbols are persisted as base vectors
        for sym in [subject, relation, obj]:
            self.store.store_symbol(sym, self.encoder.get_vector(sym))
        
        return record.memory_id

    def query(self, query_text=None, subject=None, relation=None, obj=None, top_k=5, reinforce=True):
        """Query memory with Hybrid Ranking and optional reinforcement."""
        lc = self.lifecycle if reinforce else None
        return self.retrieval.execute_query(
            subject=subject, relation=relation, obj=obj, 
            top_k=top_k, lifecycle=lc
        )

    def decay_step(self):
        """Apply a global decay step."""
        self.lifecycle.apply_decay()

    def get_context(self, current_context_keywords, top_k=5, reinforce=True):
        """Retrieve context for a given set of keywords."""
        lc = self.lifecycle if reinforce else None
        return self.retrieval.get_associative_context(
            current_context_keywords, top_k=top_k, lifecycle=lc
        )

    def consolidate(self):
        """Run the consolidation process."""
        return self.lifecycle.consolidate()

    def export_memory(self, path: str):
        """Export all memories to JSONL."""
        self.store.export_jsonl(path)

    def import_memory(self, path: str):
        """Import memories from JSONL."""
        self.store.import_jsonl(path)
        # Refresh symbol map after import
        self._load_base_vectors()
