import numpy as np
from datetime import datetime

class Ranker:
    # Pre-defined weight profiles for different retrieval needs
    PROFILES = {
        "balanced": {
            "symbolic": 0.4,
            "vector": 0.3,
            "recency": 0.15,
            "strength": 0.1,
            "provenance": 0.05
        },
        "semantic": {  # Prioritize deep HDC similarity over exact matches
            "symbolic": 0.2,
            "vector": 0.6,
            "recency": 0.05,
            "strength": 0.1,
            "provenance": 0.05
        },
        "episodic": {  # Prioritize recency and specific source/session
            "symbolic": 0.3,
            "vector": 0.2,
            "recency": 0.4,
            "strength": 0.05,
            "provenance": 0.05
        }
    }

    def __init__(self, profile="balanced"):
        self.weights = self.PROFILES.get(profile, self.PROFILES["balanced"])

    def compute_hybrid_score(self, memory_record, query_context):
        """
        Compute a multi-factor weighted score.
        query_context contains:
            - symbolic_match: bool
            - hdc_similarity: float
            - current_agent_id: str (optional)
            - preferred_source: str (optional)
        """
        scores = {}
        
        # 1. Symbolic Score (Exact match bonus)
        scores["symbolic"] = 1.0 if query_context.get("symbolic_match") else 0.0
        
        # 2. Vector Score (HDC similarity)
        scores["vector"] = max(0.0, query_context.get("hdc_similarity", 0.0))
        
        # 3. Recency Score (Exponential decay)
        try:
            created_dt = datetime.fromisoformat(memory_record.timestamp_created)
            delta_days = (datetime.now() - created_dt).total_seconds() / 86400.0
            scores["recency"] = np.exp(-delta_days / 7.0) # 7-day half-life
        except:
            scores["recency"] = 0.5
            
        # 4. Strength Score (Memory persistence)
        scores["strength"] = min(1.0, memory_record.strength)
        
        # 5. Provenance & Bias Score
        prov_score = 0.0
        if query_context.get("preferred_source") == memory_record.source:
            prov_score += 0.5
        if query_context.get("current_agent_id") == memory_record.agent_id:
            prov_score += 0.5
        scores["provenance"] = prov_score
        
        # Calculate weighted sum based on the active profile
        final_score = 0.0
        for component, weight in self.weights.items():
            final_score += scores.get(component, 0.0) * weight
            
        return final_score, scores
