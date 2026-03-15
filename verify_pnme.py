import os
import sys
# Add current directory to path
sys.path.append(os.getcwd())

from pnme.api import PNME
import numpy as np

def test_pnme():
    db_path = "test_memory.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    print("--- Initializing PNME ---")
    memory = PNME(db_path=db_path)
    
    print("\n--- Storing Memories ---")
    memory.store("user", "likes", "rust", context="discussion about programming")
    memory.store("user", "prefers", "dark_mode", context="UI settings")
    memory.store("rust", "is", "fast", context="language properties")
    
    print("\n--- Querying Memories (Symbolic + HDC) ---")
    # Query: What does user like?
    results = memory.query(subject="user", relation="likes")
    print(f"Query (user, likes, ?): {results}")
    
    # Query: Who likes rust?
    results = memory.query(relation="likes", obj="rust")
    print(f"Query (?, likes, rust): {results}")

    print("\n--- Testing Context Retrieval ---")
    # Keywords: 'programming', 'language'
    context_results = memory.retrieve_context(["programming", "fast"])
    print(f"Context results for ['programming', 'fast']:")
    for r in context_results:
        m = r['memory']
        print(f"  - {m['subject']} {m['relation']} {m['object']} (sim: {r['similarity']:.4f})")

    print("\n--- Testing Persistence ---")
    # Close and reopen
    del memory
    memory2 = PNME(db_path=db_path)
    print("Re-checking memory after restart...")
    results = memory2.query(subject="rust", relation="is")
    print(f"Query (rust, is, ?): {results}")
    
    if results and results[0]['symbol'] == 'fast':
        print("\n✅ Verification Successful: Memory persisted and recall works!")
    else:
        print("\n❌ Verification Failed: Recall incorrect.")

if __name__ == "__main__":
    test_pnme()
