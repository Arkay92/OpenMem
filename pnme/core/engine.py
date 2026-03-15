from ..hdc.encoder import HDCEncoder
from ..storage.sqlite_store import SQLiteStore
from .recall import associate_recall, find_target
from .schema import MemoryRecord
from .ranker import Ranker
from .safety import SafetyFilter
import numpy as np
from datetime import datetime

class PNMEEngine:
    def __init__(self, db_path="pnme_memory.db", dim=10000):
        self.store = SQLiteStore(db_path)
        self.encoder = HDCEncoder(dim)
        self.ranker = Ranker()
        self.safety = SafetyFilter()
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
        """
        Query memory with Hybrid Ranking and optional reinforcement.
        """
        records = self.store.get_all_records()
        memories = [r.to_dict() for r in records]
        for i, m in enumerate(memories):
            m['vector'] = records[i].vector

        final_results = []

        if subject or relation or obj:
            # Symbolic-based HDC recall
            query_ctx_v, missing_role = self.encoder.encode_query(subject, relation, obj)
            if missing_role:
                hdc_results = find_target(query_ctx_v, memories, self.encoder, missing_role,
                                          subject=subject, relation=relation, obj=obj)
                
                for res in hdc_results:
                    rec = res["source_memory"]
                    if isinstance(rec, dict):
                        rec = next((r for r in records if r.memory_id == rec['memory_id']), None)
                    
                    if rec:
                        if reinforce:
                            self._reinforce(rec)
                        
                        score, breakdown = self.ranker.compute_hybrid_score(rec, {
                            "symbolic_match": True,
                            "hdc_similarity": res["confidence"]
                        })
                        final_results.append({
                            "record": rec,
                            "score": score,
                            "explanation": breakdown,
                            "extracted_symbol": res["symbol"]
                        })

        else:
            for rec in records:
                score, breakdown = self.ranker.compute_hybrid_score(rec, {
                    "symbolic_match": False,
                    "hdc_similarity": 0.0
                })
                final_results.append({
                    "record": rec,
                    "score": score,
                    "explanation": breakdown
                })

        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:top_k]

    def _reinforce(self, record: MemoryRecord):
        """Reinforce a memory record on access."""
        new_count = record.reinforcement_count + 1
        # Strength increases asymptotically towards 1.0 (or slightly above if supported)
        new_strength = min(1.5, record.strength + 0.1 / (1 + record.reinforcement_count*0.1))
        
        self.store.update_memory_metadata(record.memory_id, {
            "reinforcement_count": new_count,
            "strength": new_strength,
            "timestamp_last_accessed": datetime.now().isoformat()
        })
        # Update local object too if possible
        record.reinforcement_count = new_count
        record.strength = new_strength
        record.timestamp_last_accessed = datetime.now().isoformat()

    def decay_step(self):
        """Apply a global decay step to all memories."""
        records = self.store.get_all_records()
        for rec in records:
            # Strength decreases based on its decay_factor
            new_strength = max(0.01, rec.strength - rec.decay_factor)
            self.store.update_memory_metadata(rec.memory_id, {"strength": new_strength})

    def get_context(self, current_context_keywords, top_k=5, reinforce=True):
        """Retrieve relevant memories for a given set of keywords with ranking."""
        keyword_vectors = [self.encoder.get_vector(k) for k in current_context_keywords]
        if not keyword_vectors:
            return []
            
        from ..hdc.ops import bundle
        context_v = bundle(keyword_vectors)
        
        records = self.store.get_all_records()
        memories = [r.to_dict() for r in records]
        for i, m in enumerate(memories):
            m['vector'] = records[i].vector
            
        hdc_results = associate_recall(context_v, memories)
        
        final_results = []
        for res in hdc_results:
            rec_dict = res["memory"]
            rec = next((r for r in records if r.memory_id == rec_dict['memory_id']), None)
            if rec:
                if reinforce:
                    self._reinforce(rec)
                
                score, breakdown = self.ranker.compute_hybrid_score(rec, {
                    "symbolic_match": False,
                    "hdc_similarity": res["similarity"]
                })
                final_results.append({
                    "record": rec,
                    "score": score,
                    "explanation": breakdown
                })
        
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:top_k]

    def consolidate(self):
        """
        Consolidation process:
        1. Promote episodic memories with high reinforcement to 'semantic'.
        """
        records = self.store.get_all_records()
        promoted_count = 0
        for rec in records:
            if rec.memory_type == "episodic" and rec.reinforcement_count >= 3:
                self.store.update_memory_metadata(rec.memory_id, {
                    "memory_type": "semantic",
                    "strength": min(1.5, rec.strength + 0.2)
                })
                promoted_count += 1
        return promoted_count
