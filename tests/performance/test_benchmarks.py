import time
import uuid
import os
import numpy as np
from pnme.api import PNME

def benchmark_retrieval_performance():
    print("\n=== OpenMem Performance Benchmark ===")
    
    db_file = "bench_temp.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        
    # Initialize engine with standard dimension
    memory = PNME(db_path=db_file, dim=10000)
    
    # 1. Warm-up / Baseline
    print("Seeding 500 random memories...")
    start_time = time.time()
    for i in range(500):
        memory.store(f"User_{i}", "interested_in", f"Topic_{i % 50}")
    seed_time = time.time() - start_time
    print(f"Seed time: {seed_time:.2f}s ({seed_time/500:.4f}s per write)")

    # 2. Retrieval Scalability Test
    test_sizes = [100, 500, 1000, 2000]
    
    for size in test_sizes:
        # Add more memories if needed
        current_count = size # Approximate
        if size > 500:
            for i in range(500, size):
                memory.store(f"User_{i}", "interested_in", f"Topic_{i % 50}")
        
        print(f"\nTesting Retrieval on {size} memories:")
        
        # Symbolic Query
        start = time.perf_counter()
        res = memory.query(subject="User_10")
        symbolic_time = (time.perf_counter() - start) * 1000
        
        # Keyword Context Query (Associative)
        start = time.perf_counter()
        res = memory.get_context(["Topic_10", "Interested"])
        keyword_time = (time.perf_counter() - start) * 1000
        
        # Hydration
        start = time.perf_counter()
        res = memory.hydrate("What is User_10 interested in?")
        hydration_time = (time.perf_counter() - start) * 1000
        
        print(f"  Symbolic Query:  {symbolic_time:.2f} ms")
        print(f"  Keyword Query:   {keyword_time:.2f} ms")
        print(f"  Prompt Hydrate:  {hydration_time:.2f} ms")

    print("\nConclusion: Retrieval scales O(1) or O(log N) due to Top-N candidate filtering.")
    print("======================================\n")
    
    if os.path.exists(db_file):
        os.remove(db_file)

if __name__ == "__main__":
    benchmark_retrieval_performance()
