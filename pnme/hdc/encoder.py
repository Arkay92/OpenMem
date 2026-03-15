import numpy as np
from .ops import create_vector, bind, bundle, permute

class HDCEncoder:
    def __init__(self, dim=10000):
        self.dim = dim
        self.symbol_map = {}
        # Pre-defined Role Vectors (using deterministic seeds)
        self.role_subject = create_vector(dim, seed=101)
        self.role_relation = create_vector(dim, seed=102)
        self.role_object = create_vector(dim, seed=103)
        self.role_context = create_vector(dim, seed=104)

    def get_vector(self, symbol):
        """Retrieve or create a base vector for a symbol."""
        if symbol not in self.symbol_map:
            # Deterministic generation based on symbol name hashing could be an improvement
            # but for now we rely on the internal map.
            self.symbol_map[symbol] = create_vector(self.dim)
        return self.symbol_map[symbol]

    def encode_triple(self, subject, relation, object_val, context=None):
        """
        Encode a semantic triple into a single HDC vector using Role Vectors.
        Formula: bundle([bind(R_s, S), bind(R_r, R), bind(R_o, O)])
        """
        v_s = bind(self.role_subject, self.get_vector(subject))
        v_r = bind(self.role_relation, self.get_vector(relation))
        v_o = bind(self.role_object, self.get_vector(object_val))
        
        vectors = [v_s, v_r, v_o]
        if context:
            v_c = bind(self.role_context, self.get_vector(context))
            vectors.append(v_c)
            
        return bundle(vectors)

    def encode_query(self, subject=None, relation=None, object_val=None):
        """
        Produce a query context vector with one missing part.
        In the bundle approach, we use the role vector of the missing part to extract it.
        Example: To find ?, we start with the Memory vector.
        Actually, for bundle-based memory, recall usually involves similarity against the whole bundle
        or partial unbinding.
        """
        # For compatibility with existing find_target:
        # If we use XOR/Bind-based triple: bind(p(s), R, O)
        # Roles were: p(s) for subject, ID for relation, ID for object.
        # Transitioning to the bundle-based approach requires updating recall logic.
        
        if subject is None:
            return None, "subject" # Signal role-based retrieval
        if relation is None:
            return None, "relation"
        if object_val is None:
            return None, "object"
        
        return None, None
