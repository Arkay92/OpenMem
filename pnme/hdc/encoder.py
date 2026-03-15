import numpy as np
import hashlib
from .ops import create_vector, bind, bundle, unbind

class HDCEncoder:
    def __init__(self, dim=10000):
        self.dim = dim
        self.symbol_map = {}
        # Pre-defined Role Vectors (using deterministic seeds)
        self.role_subject = create_vector(dim, seed=101)
        self.role_relation = create_vector(dim, seed=102)
        self.role_object = create_vector(dim, seed=103)
        self.role_context = create_vector(dim, seed=104)

    def _get_deterministic_seed(self, symbol):
        """Generate a deterministic 32-bit integer seed from a symbol string."""
        h = hashlib.sha256(symbol.encode()).digest()
        return int.from_bytes(h[:4], "big")

    def get_vector(self, symbol):
        """Retrieve or create a base vector for a symbol using hash-based determinism."""
        if symbol not in self.symbol_map:
            seed = self._get_deterministic_seed(symbol)
            self.symbol_map[symbol] = create_vector(self.dim, seed=seed)
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

    def encode_query(self, subject=None, relation=None, object_val=None, context=None):
        """
        Produce a partial-query bundle vector and identify missing roles.
        Returns: (query_vector, missing_roles_list)
        """
        known = []
        missing = []

        if subject is None:
            missing.append("subject")
        else:
            known.append(bind(self.role_subject, self.get_vector(subject)))

        if relation is None:
            missing.append("relation")
        else:
            known.append(bind(self.role_relation, self.get_vector(relation)))

        if object_val is None:
            missing.append("object")
        else:
            known.append(bind(self.role_object, self.get_vector(object_val)))

        if context is not None:
            known.append(bind(self.role_context, self.get_vector(context)))

        query_v = bundle(known) if known else None
        return query_v, missing
