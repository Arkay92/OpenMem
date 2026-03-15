import numpy as np
from ..hdc.ops import similarity, unbind

def associate_recall(query_v, memories, top_k=5):
    """
    Given a query vector (partial bundle), find the most similar memories.
    This implements 'bundle mode' retrieval.
    """
    if query_v is None:
        return []
        
    results = []
    for mem in memories:
        # Support both dict and object
        vector = mem["vector"] if isinstance(mem, dict) else mem.vector
        sim = similarity(query_v, vector)
        results.append({
            "memory": mem,
            "similarity": sim
        })
    
    # Sort by similarity descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

def find_target(query_v, memories, encoder, missing_role, subject=None, relation=None, obj=None, top_k=5):
    """
    Implement 'role-unbinding mode' retrieval.
    Given a partial query and memories, unbind the missing role to find candidates.
    """
    candidates = []
    
    # Identify the target role vector for unbinding
    if missing_role == "subject":
        role_v = encoder.role_subject
    elif missing_role == "relation":
        role_v = encoder.role_relation
    elif missing_role == "object":
        role_v = encoder.role_object
    else:
        # Fallback to associate recall if role is ambiguous or context-only
        results = associate_recall(query_v, memories, top_k=top_k)
        # Map keys to match find_target's expected output
        return [
            {
                "symbol": None,
                "confidence": r["similarity"],
                "source_memory": r["memory"]
            }
            for r in results
        ]

    # For each memory, optionally filter symbolically, then unbind
    for mem in memories:
        # Symbolic Filtering (Hybrid Retrieval)
        if subject and mem.get('subject') != subject: continue
        if relation and mem.get('relation') != relation: continue
        if obj and mem.get('object') != obj: continue

        vector = mem["vector"] if isinstance(mem, dict) else mem.vector
        
        # Role-Unbinding: target_v = unbind(Memory, Role_Missing)
        extracted_v = unbind(vector, role_v)
        
        # Match against known symbols in the encoder's cache
        best_sym = None
        max_sim = -1.0
        for sym, base_v in encoder.symbol_map.items():
            sim = similarity(extracted_v, base_v)
            if sim > max_sim:
                max_sim = sim
                best_sym = sym
        
        # If the query provided a partial match, we can also score the memory similarity
        # to the partial query bundle itself.
        match_sim = similarity(query_v, vector) if query_v is not None else 1.0
        
        if max_sim > 0.1:
            candidates.append({
                "symbol": best_sym,
                "confidence": max_sim * match_sim, # Combined confidence
                "source_memory": mem
            })
            
    # Sort by confidence
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates[:top_k]
