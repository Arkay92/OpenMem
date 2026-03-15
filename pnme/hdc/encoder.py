import numpy as np
from .ops import create_vector, bind, permute

class HDCEncoder:
    def __init__(self, dim=10000):
        self.dim = dim
        self.symbol_map = {}

    def get_vector(self, symbol):
        """Retrieve or create a base vector for a symbol."""
        if symbol not in self.symbol_map:
            self.symbol_map[symbol] = create_vector(self.dim)
        return self.symbol_map[symbol]

    def encode_triple(self, subject, relation, object_val):
        """
        Encode a semantic triple into a single HDC vector.
        Formula: permute(subject_v, 1) ^ relation_v ^ object_v
        """
        v_s = self.get_vector(subject)
        v_r = self.get_vector(relation)
        v_o = self.get_vector(object_val)
        
        # We use permute(subject) to distinguish roles
        return bind(bind(permute(v_s, 1), v_r), v_o)

    def encode_query(self, subject=None, relation=None, object_val=None):
        """
        Produce a query vector with one missing part.
        Returns the query vector and the part that is missing.
        """
        if subject is None:
            # Query: (? ^ relation ^ object)
            # To find ?, we bind the query vector with (relation ^ object)
            # Actually, to find subject, we need the stored memory vector bound with (relation ^ object)
            # This is handled in the recall logic, but here we provide the "context" vector.
            return bind(self.get_vector(relation), self.get_vector(object_val)), "subject"
        
        if relation is None:
            return bind(permute(self.get_vector(subject), 1), self.get_vector(object_val)), "relation"
        
        if object_val is None:
            return bind(permute(self.get_vector(subject), 1), self.get_vector(relation)), "object"
        
        return None, None
