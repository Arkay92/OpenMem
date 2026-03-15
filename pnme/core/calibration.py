import numpy as np
from typing import List, Dict, Any
from ..hdc.ops import similarity

class Calibrator:
    """
    Utility for monitoring PNME performance and calibrating HDC thresholds.
    """
    def __init__(self, engine):
        self.engine = engine

    def get_stats(self) -> Dict[str, Any]:
        """Gather database and memory statistics."""
        records = self.engine.store.get_all_records()
        symbols = self.engine.store.load_symbols()
        
        return {
            "total_memories": len(records),
            "total_symbols": len(symbols),
            "memory_types": self._count_types(records),
            "avg_strength": np.mean([r.strength for r in records]) if records else 0,
            "db_size_bytes": self._get_db_size()
        }

    def _count_types(self, records: List[Any]) -> Dict[str, int]:
        types = {}
        for r in records:
            t = r.memory_type
            types[t] = types.get(t, 0) + 1
        return types

    def _get_db_size(self) -> int:
        import os
        if os.path.exists(self.engine.store.db_path):
            return os.path.getsize(self.engine.store.db_path)
        return 0

    def evaluate_recall_health(self) -> Dict[str, float]:
        """
        Evaluate the quality of associative recall.
        Calculates the signal-to-noise ratio in HDC extraction.
        """
        records = self.engine.store.get_all_records()
        if not records:
            return {"avg_noise": 0.0, "avg_signal": 0.0}
            
        signals = []
        for rec in records:
            # We know the answer for this record. How clear is it in the vector?
            # M = bind(Rs, S) + bind(Rr, R) + bind(Ro, O)
            # Project using Rs should give S
            extracted_v = rec.vector * self.engine.encoder.role_subject
            target_v = self.engine.encoder.get_vector(rec.subject)
            signals.append(similarity(extracted_v, target_v))
            
        return {
            "avg_retrieval_signal": float(np.mean(signals)),
            "suggested_threshold": float(np.mean(signals) * 0.7)
        }
