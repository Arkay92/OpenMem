from datetime import datetime
from .schema import MemoryRecord

class MemoryLifecycle:
    def __init__(self, store):
        self.store = store

    def reinforce(self, record: MemoryRecord):
        """Reinforce a memory record on access."""
        new_count = record.reinforcement_count + 1
        # Strength increases asymptotically
        new_strength = min(1.5, record.strength + 0.1 / (1 + record.reinforcement_count * 0.1))
        
        updates = {
            "reinforcement_count": new_count,
            "strength": new_strength,
            "timestamp_last_accessed": datetime.now().isoformat()
        }
        self.store.update_memory_metadata(record.memory_id, updates)
        
        # Update local object
        record.reinforcement_count = new_count
        record.strength = new_strength
        record.timestamp_last_accessed = updates["timestamp_last_accessed"]

    def apply_decay(self):
        """Apply a global decay step based on current decay factors."""
        records = self.store.get_all_records()
        for rec in records:
            new_strength = max(0.01, rec.strength - rec.decay_factor)
            self.store.update_memory_metadata(rec.memory_id, {"strength": new_strength})

    def consolidate(self):
        """
        Promote episodic memories to semantic if they have high reinforcement.
        """
        records = self.store.get_all_records()
        promoted_count = 0
        for rec in records:
            if rec.memory_type == "episodic" and rec.reinforcement_count >= 3:
                self.store.update_memory_metadata(rec.memory_id, {
                    "memory_type": "semantic",
                    "strength": min(1.5, rec.strength + 0.2)
                })
                promoted_count += 1
        return promoted_count
