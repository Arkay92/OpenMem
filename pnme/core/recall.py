import numpy as np
from ..hdc.ops import similarity, unbind

def associate_recall(query_v, memories, top_k=5):
    """
    Given a query vector (partial bundle), find the most similar memories.
    Uses vectorized NumPy operations for O(N) efficiency.
    """
    if query_v is None or not memories:
        return []
        
    # Vectorized similarity (Stage 14 optimization)
    mem_vectors = np.stack([m["vector"] if isinstance(m, dict) else m.vector for m in memories])
    # dot_product similarity for bipolar vectors
    dot_products = np.dot(mem_vectors, query_v.astype(np.float64))
    # dim = query_v.shape[0]
    similarities = dot_products / (np.sqrt(query_v.shape[0]) * np.sqrt(query_v.shape[0]))
    
    indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in indices:
        results.append({
            "memory": memories[idx],
            "similarity": similarities[idx]
        })
    return results

def find_target(query_v, memories, encoder, missing_roles, subject=None, relation=None, obj=None, top_k=5):
    """
    Implement 'role-unbinding mode' retrieval for one or more missing roles.
    Optimized with candidate filtering (Stage 14).
    """
    if not memories:
        return []
    if isinstance(missing_roles, str):
        missing_roles = [missing_roles]
        
    # 1. Symbolic + Broad Vector Filter (Stage 14 optimization)
    # Perform fast vectorized similarity check on the bundle first
    mem_vectors = np.stack([m["vector"] if isinstance(m, dict) else m.vector for m in memories])
    
    # Bundle similarity (broad scan)
    if query_v is not None:
        dot_products = np.dot(mem_vectors, query_v.astype(np.float64))
        bundle_sims = dot_products / (encoder.dim) # Bipolar vectors norm is sqrt(dim)
    else:
        bundle_sims = np.ones(len(memories))

    # Pre-select candidates based on symbolic constraints AND bundle similarity
    # We take top_k * 5 or 20, whichever is larger, as the candidate pool for expensive unbinding
    candidate_indices = []
    for i, mem in enumerate(memories):
        # Symbolic Filtering
        if subject and mem.get('subject') != subject: continue
        if relation and mem.get('relation') != relation: continue
        if obj and mem.get('object') != obj: continue
        candidate_indices.append((i, bundle_sims[i]))
    
    # Sort candidates by bundle similarity and take top pool
    candidate_indices.sort(key=lambda x: x[1], reverse=True)
    pool_indices = [idx for idx, sim in candidate_indices[:max(20, top_k * 4)]]
    
    candidates = []
    role_map = {
        "subject": encoder.role_subject,
        "relation": encoder.role_relation,
        "object": encoder.role_object,
        "context": encoder.role_context
    }

    # 2. Expensive Unbinding on Candidate Pool
    for idx in pool_indices:
        mem = memories[idx]
        vector = mem["vector"] if isinstance(mem, dict) else mem.vector
        bundle_sim = bundle_sims[idx]
        
        extracted_results = {}
        total_max_sim = 0.0
        
        for role in missing_roles:
            role_v = role_map.get(role)
            if role_v is None: continue
            
            target_v = unbind(vector, role_v)
            
            # Vectorized Symbol Match (already optimized in Stage 13)
            if encoder.symbol_map:
                symbols = list(encoder.symbol_map.keys())
                symbol_matrix = np.stack(list(encoder.symbol_map.values()))
                dot_products = np.dot(symbol_matrix, target_v.astype(np.float64))
                similarities = dot_products / (encoder.dim)
                
                match_idx = np.argmax(similarities)
                max_sim = similarities[match_idx]
                best_sym = symbols[match_idx]
            else:
                best_sym = None
                max_sim = -1.0
            
            extracted_results[role] = {"symbol": best_sym, "confidence": max_sim}
            total_max_sim += max_sim
        
        if not missing_roles:
            final_conf = bundle_sim
        else:
            avg_conf = total_max_sim / len(missing_roles)
            final_conf = avg_conf * (0.5 + 0.5 * bundle_sim) # Blend unbinding conf with bundle sim

        if final_conf > 0.05:
            candidates.append({
                "extracted_symbols": extracted_results,
                "confidence": final_conf,
                "source_memory": mem
            })
            
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates[:top_k]
