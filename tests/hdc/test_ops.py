import numpy as np
import pytest
from pnme.hdc.ops import create_vector, bind, bundle, similarity

def test_create_vector():
    v = create_vector(1000, seed=42)
    assert v.shape == (1000,)
    assert np.all(np.abs(v) == 1.0)
    
    # Check determinism
    v2 = create_vector(1000, seed=42)
    assert np.array_equal(v, v2)

def test_bind():
    v1 = create_vector(1000, seed=1)
    v2 = create_vector(1000, seed=2)
    bound = bind(v1, v2)
    assert bound.shape == (1000,)
    assert np.all(np.abs(bound) == 1.0)
    # Binding is commutative (XOR or multiplication of BIP)
    assert np.array_equal(bind(v1, v2), bind(v2, v1))

def test_bundle():
    v1 = create_vector(1000, seed=1)
    v2 = create_vector(1000, seed=2)
    v3 = create_vector(1000, seed=3)
    bundled = bundle([v1, v2, v3])
    # bundle should be normalized to +/- 1
    assert bundled.shape == (1000,)
    assert np.all(np.abs(bundled) == 1.0)

def test_similarity():
    v1 = create_vector(1000, seed=1)
    assert similarity(v1, v1) == 1.0
    
    v2 = create_vector(1000, seed=2)
    # Random vectors should have near 0 similarity
    assert abs(similarity(v1, v2)) < 0.1
