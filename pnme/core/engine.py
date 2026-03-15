from ..hdc.encoder import HDCEncoder
from ..storage.sqlite_store import SQLiteStore
from .recall import associate_recall, find_target
import numpy as np

class PNMEEngine:
    def __init__(self, db_path="pnme_memory.db", dim=10000):
        self.store = SQLiteStore(db_path)
        self.encoder = HDCEncoder(dim)
        self._load_base_vectors()

    def _load_base_vectors(self):
        """Restore symbol map from storage."""
        symbols = self.store.load_symbols()
        self.encoder.symbol_map.update(symbols)

    def write(self, subject, relation, obj, context=""):
        """Store a semantic memory."""
        # Encode
        vector = self.encoder.encode_triple(subject, relation, obj)
        
        # Save memory
        self.store.store_memory(subject, relation, obj, context, vector)
        
        # Ensure all involved symbols are persisted as base vectors
        for sym in [subject, relation, obj]:
            self.store.store_symbol(sym, self.encoder.get_vector(sym))
        
        return True

    def query(self, query_text=None, subject=None, relation=None, obj=None):
        """
        Query memory. 
        If symbolic parts are provided, it performs specific associative recall.
        TODO: Natural language query integration (needs LLM pass or semantic embedding).
        """
        memories = self.store.get_all_memories()
        
        if subject or relation or obj:
            # Symbolic-based HDC recall
            query_ctx_v, missing_role = self.encoder.encode_query(subject, relation, obj)
            if query_ctx_v is not None:
                return find_target(query_ctx_v, memories, self.encoder, missing_role)

        # General associative recall (similarity search)
        # If we have a partially filled triple, we can construct a search vector
        # For now, return all memories if no specific query
        return memories

    def get_context(self, current_context_keywords):
        """Retrieve relevant memories for a given set of keywords."""
        # Bundle keyword vectors to create a context vector
        keyword_vectors = [self.encoder.get_vector(k) for k in current_context_keywords]
        if not keyword_vectors:
            return []
            
        from ..hdc.ops import bundle
        context_v = bundle(keyword_vectors)
        
        memories = self.store.get_all_memories()
        return associate_recall(context_v, memories)
