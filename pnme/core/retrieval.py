from .recall import associate_recall, find_target
from ..hdc.ops import bundle

class RetrievalPipeline:
    def __init__(self, store, encoder, ranker):
        self.store = store
        self.encoder = encoder
        self.ranker = ranker

    def execute_query(self, query_text=None, subject=None, relation=None, obj=None, top_k=5, lifecycle=None):
        """Execute a hybrid query utilizing symbolic, HDC, and text-based keywords."""
        records = self.store.get_all_records()
        memories = [r.to_dict() for r in records]
        
        # 1. Broad Text Selection (if query_text exists)
        text_boosts = {}
        if query_text:
            import re
            keywords = re.findall(r'\w+', query_text.lower())
            if keywords:
                assoc_results = self.get_associative_context(keywords, top_k=top_k*2)
                for res in assoc_results:
                    text_boosts[res["record"].memory_id] = res["score"]

        final_results = []
        
        # 2. Hybrid Search (Symbolic/HDC/Text)
        if subject or relation or obj:
            # Symbolic-based HDC recall
            query_ctx_v, missing_roles = self.encoder.encode_query(subject, relation, obj)
            
            # Optimized find_target (uses Stage 14 candidate filtering)
            hdc_results = find_target(query_ctx_v, memories, self.encoder, missing_roles,
                                      subject=subject, relation=relation, obj=obj, top_k=top_k*2)
            
            for res in hdc_results:
                rec_dict = res["source_memory"]
                rec = next((r for r in records if r.memory_id == rec_dict['memory_id']), None)
                
                if rec:
                    if lifecycle:
                        lifecycle.reinforce(rec)
                    
                    self.store.log_access(rec.memory_id, query_type="hybrid")
                    
                    boost = text_boosts.get(rec.memory_id, 0.0)
                    score, breakdown = self.ranker.compute_hybrid_score(rec, {
                        "symbolic_match": True,
                        "hdc_similarity": res["confidence"],
                        "text_boost": boost
                    })
                    
                    final_results.append({
                        "record": rec,
                        "score": score,
                        "explanation": breakdown,
                        "extracted_symbols": res.get("extracted_symbols")
                    })
        elif query_text:
            # Pure text-based associative query
            return self.get_associative_context(keywords, top_k=top_k, lifecycle=lifecycle)
        else:
            # Broad scan / No constraints
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
                self.store.log_access(rec.memory_id, query_type="full_scan")

        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:top_k]

    def get_associative_context(self, keywords, top_k=5, lifecycle=None):
        """Retrieve memories based on keyword association."""
        keyword_vectors = [self.encoder.get_vector(k) for k in keywords]
        if not keyword_vectors:
            return []
            
        context_v = bundle(keyword_vectors)
        records = self.store.get_all_records()
        memories = [r.to_dict() for r in records]
            
        hdc_results = associate_recall(context_v, memories)
        
        final_results = []
        for res in hdc_results:
            rec_dict = res["source_memory"] if "source_memory" in res else res.get("memory")
            rec = next((r for r in records if r.memory_id == rec_dict['memory_id']), None)
            if rec:
                if lifecycle:
                    lifecycle.reinforce(rec)
                
                self.store.log_access(rec.memory_id, query_type="associative")
                
                score, breakdown = self.ranker.compute_hybrid_score(rec, {
                    "symbolic_match": False,
                    "hdc_similarity": res.get("similarity", res.get("confidence", 0.0))
                })
                final_results.append({
                    "record": rec,
                    "score": score,
                    "explanation": breakdown
                })
        
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:top_k]
