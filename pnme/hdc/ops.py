import numpy as np

def create_vector(dim=10000, seed=None):
    """Create a random bipolar vector (-1, 1)."""
    if seed is not None:
        rng = np.random.default_rng(seed)
        return rng.choice([-1, 1], size=dim).astype(np.int8)
    return np.random.choice([-1, 1], size=dim).astype(np.int8)

def bind(v1, v2):
    """Binding operation: element-wise multiplication (XOR in bipolar space)."""
    return v1 * v2

def unbind(v1, v2):
    """Unbinding operation: identical to bind for bipolar vectors (v1 * v2)."""
    return v1 * v2

def bundle(vectors):
    """Bundling operation: element-wise sum followed by thresholding."""
    v_sum = np.sum(vectors, axis=0)
    # Threshold at 0. Mapping 0 to +1 for determinism (Stage 13 hardening)
    bundled = np.where(v_sum >= 0, 1, -1).astype(np.int8)
    return bundled

def permute(v, shift=1):
    """Permutation operation: circular shift."""
    return np.roll(v, shift)

def similarity(v1, v2):
    """Cosine similarity between two bipolar vectors."""
    v1_f = v1.astype(np.float64)
    v2_f = v2.astype(np.float64)
    return np.dot(v1_f, v2_f) / (np.linalg.norm(v1_f) * np.linalg.norm(v2_f))
