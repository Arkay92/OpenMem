import numpy as np
from ..hdc.ops import similarity

def associate_recall(query_v, memories, top_k=5):
    """
    Given a query vector (partial information binded), find the most similar memories.
    Returns a list of potential candidates with similarity scores.
    """
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

def find_target(query_context_v, memories, encoder, missing_role, subject=None, relation=None, obj=None, top_k=5):
    """
    Specifically for role-based retrieval with Hybrid Symbolic Filtering.
    """
    candidates = []
    symbol_vectors = encoder.symbol_map
    
    # Determine which role vector to use for projection
    if missing_role == "subject":
        role_v = encoder.role_subject
    elif missing_role == "relation":
        role_v = encoder.role_relation
    elif missing_role == "object":
        role_v = encoder.role_object
    else:
        return []

    # For each memory, check symbolic constraints if provided
    for mem in memories:
        # Hybrid Filter: Only process memories that match the known symbolic parts
        # If subject is provided, memory's subject must match
        if subject and mem.get('subject') != subject: continue
        if relation and mem.get('relation') != relation: continue
        if obj and mem.get('object') != obj: continue

        vector = mem["vector"] if isinstance(mem, dict) else mem.vector
        
        # Project: extracted_v = bind(M, Role_Target)
        extracted_v = vector * role_v
        
        # Check similarity with all base symbols
        best_sym = None
        max_sim = -1.0
        for sym, base_v in symbol_vectors.items():
            sim = similarity(extracted_v, base_v)
            if sim > max_sim:
                max_sim = sim
                best_sym = sym
        
        if max_sim > 0.1:
            candidates.append({
                "symbol": best_sym,
                "confidence": max_sim,
                "source_memory": mem
            })
            
    # Deduplicate and take top-k
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates[:top_k]
