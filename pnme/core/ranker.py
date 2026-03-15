import numpy as np
from datetime import datetime

class Ranker:
    def __init__(self, weights=None):
        self.weights = weights or {
            "symbolic": 0.4,
            "vector": 0.3,
            "recency": 0.2,
            "strength": 0.1
        }

    def compute_hybrid_score(self, memory_record, query_context):
        """
        Compute a weighted score for a memory record relative to a query.
        query_context contains:
            - symbolic_match: bool
            - hdc_similarity: float
        """
        scores = {}
        
        # 1. Symbolic Score (Exact match bonus)
        scores["symbolic"] = 1.0 if query_context.get("symbolic_match") else 0.0
        
        # 2. Vector Score (HDC similarity)
        scores["vector"] = max(0.0, query_context.get("hdc_similarity", 0.0))
        
        # 3. Recency Score (Exponential decay)
        # Assuming timestamp_created is ISO format
        try:
            created_dt = datetime.fromisoformat(memory_record.timestamp_created)
            delta_days = (datetime.now() - created_dt).total_seconds() / 86400.0
            scores["recency"] = np.exp(-delta_days / 7.0) # Decay over 7 days half-life roughly
        except:
            scores["recency"] = 0.5
            
        # 4. Strength Score (Normalized reinforcement)
        # memory_record.strength is already a float
        scores["strength"] = min(1.0, memory_record.strength)
        
        # Final weighted sum
        final_score = sum(scores[k] * self.weights[k] for k in self.weights)
        
        return final_score, scores
