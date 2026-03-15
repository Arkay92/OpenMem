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

def find_target(query_v, memories, encoder, missing_roles, subject=None, relation=None, obj=None, top_k=5):
    """
    Implement 'role-unbinding mode' retrieval for one or more missing roles.
    """
    if isinstance(missing_roles, str):
        missing_roles = [missing_roles]
        
    candidates = []
    
    # Map role names to role vectors
    role_map = {
        "subject": encoder.role_subject,
        "relation": encoder.role_relation,
        "object": encoder.role_object,
        "context": encoder.role_context
    }

    # For each memory, optionally filter symbolically, then unbind all missing roles
    for mem in memories:
        # Symbolic Filtering (Hybrid Retrieval)
        if subject and mem.get('subject') != subject: continue
        if relation and mem.get('relation') != relation: continue
        if obj and mem.get('object') != obj: continue

        vector = mem["vector"] if isinstance(mem, dict) else mem.vector
        
        extracted_results = {}
        total_max_sim = 0.0
        
        for role in missing_roles:
            role_v = role_map.get(role)
            if role_v is None: continue
            
            # Role-Unbinding: candidate_v = unbind(Memory, Role_Missing)
            target_v = unbind(vector, role_v)
            
            # Match against known symbols
            best_sym = None
            max_sim = -1.0
            for sym, base_v in encoder.symbol_map.items():
                sim = similarity(target_v, base_v)
                if sim > max_sim:
                    max_sim = sim
                    best_sym = sym
            
            extracted_results[role] = {"symbol": best_sym, "confidence": max_sim}
            total_max_sim += max_sim
        
        # If no missing roles were processed (empty query or something), use associate_recall
        if not missing_roles:
            match_sim = similarity(query_v, vector) if query_v is not None else 1.0
            total_max_sim = match_sim
        else:
            # Average confidence across missing roles, then combine with query match sim
            avg_conf = total_max_sim / len(missing_roles)
            match_sim = similarity(query_v, vector) if query_v is not None else 1.0
            total_max_sim = avg_conf * match_sim

        if total_max_sim > 0.05:
            candidates.append({
                "extracted_symbols": extracted_results,
                "confidence": total_max_sim,
                "source_memory": mem
            })
            
    # Sort by confidence
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates[:top_k]
