import numpy as np
from ..hdc.ops import similarity

def associate_recall(query_v, memories, top_k=5):
    """
    Given a query vector (partial information binded), find the most similar memories.
    Returns a list of potential candidates with similarity scores.
    """
    results = []
    for mem in memories:
        sim = similarity(query_v, mem["vector"])
        results.append({
            "memory": mem,
            "similarity": sim
        })
    
    # Sort by similarity descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

def find_target(query_context_v, memories, encoder, missing_role, top_k=5):
    """
    Specifically for (S ⊗ R ⊗ ?) queries.
    We bind the memory vector with (S ⊗ R) to extract ?.
    Then we compare the extracted vector against all known base vectors to find the best object candidate.
    """
    # This is slightly different from associate_recall above.
    # Associate_recall finds the *memory record* itself.
    # This function attempts to find the *symbol* that fills the gap.
    
    # Extract candidate vector for the missing role
    # memory_v = S ^ R ^ O -> O = memory_v ^ S ^ R
    # query_context_v is already (S ^ R) or (S ^ O) or (R ^ O)
    
    candidates = []
    symbol_vectors = encoder.symbol_map
    
    # For each memory, try to extract the missing piece and see what base symbol it's closest to
    for mem in memories:
        # Extracted piece
        extracted_v = mem["vector"] * query_context_v # Binding with result of bind(p(s), r)
        
        # Check similarity with all base symbols
        best_sym = None
        max_sim = -1.0
        for sym, base_v in symbol_vectors.items():
            # If we are looking for subject, we must permute the base vector to match the stored role
            target_v = base_v
            if missing_role == "subject":
                from ..hdc.ops import permute
                target_v = permute(base_v, 1)
                
            sim = similarity(extracted_v, target_v)
            if sim > max_sim:
                max_sim = sim
                best_sym = sym
        
        if max_sim > 0.5: # Threshold for confidence
            candidates.append({
                "symbol": best_sym,
                "confidence": max_sim,
                "source_memory": mem
            })
            
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates[:top_k]
