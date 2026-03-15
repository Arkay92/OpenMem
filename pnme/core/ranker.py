import numpy as np
from datetime import datetime

class Ranker:
    # Pre-defined weight profiles for different retrieval needs
    PROFILES = {
        "balanced": {
            "symbolic": 0.35,
            "vector": 0.25,
            "recency": 0.1,
            "strength": 0.05,
            "provenance": 0.05,
            "text_boost": 0.15,
            "confidence": 0.05
        },
        "semantic": {
            "symbolic": 0.15,
            "vector": 0.5,
            "recency": 0.05,
            "strength": 0.1,
            "provenance": 0.05,
            "text_boost": 0.1,
            "confidence": 0.05
        },
        "episodic": {
            "symbolic": 0.25,
            "vector": 0.15,
            "recency": 0.35,
            "strength": 0.05,
            "provenance": 0.1,
            "text_boost": 0.05,
            "confidence": 0.05
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
            - text_boost: float (Stage 15)
            - current_agent_id: str (optional)
            - preferred_source: str (optional)
            - privacy_filter: int (optional)
            - tags: List[str] (optional)
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
        
        # Tag match bonus
        query_tags = query_context.get("tags", [])
        if query_tags and memory_record.tags:
            match_count = len(set(query_tags) & set(memory_record.tags))
            prov_score += 0.2 * match_count
            
        scores["provenance"] = min(1.0, prov_score)

        # 6. Text Boost (Stage 15 natural language relevance)
        scores["text_boost"] = query_context.get("text_boost", 0.0)

        # 7. Confidence (Record-level evidence strength)
        scores["confidence"] = memory_record.confidence

        # Privacy Filter (Hard Constraint or Penalty)
        privacy_limit = query_context.get("privacy_filter", 100)
        if memory_record.privacy_level > privacy_limit:
            return 0.0, {"blocked": "privacy_violation"}
        
        # Calculate weighted sum based on the active profile
        final_score = 0.0
        for component, weight in self.weights.items():
            final_score += scores.get(component, 0.0) * weight
            
        return final_score, scores
