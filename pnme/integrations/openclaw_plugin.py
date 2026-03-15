from ..api import PNME

class PNMEPlugin:
    """OpenClaw Plugin for Persistent Neuro-Symbolic Memory."""
    def __init__(self, db_path="pnme_memory.db"):
        self.pnme = PNME(db_path)
        self.name = "persistent_memory"

    def get_skills(self):
        """Register skills for OpenClaw."""
        return {
            "store_memory": self.store_memory,
            "query_memory": self.query_memory,
            "recall_associations": self.recall_associations,
            "get_context": self.get_context
        }

    def store_memory(self, subject, relation, object_val, context=""):
        """Store a fact."""
        return self.pnme.store(subject, relation, object_val, context=context)

    def query_memory(self, subject=None, relation=None, object_val=None, top_k=5):
        """Query facts."""
        return self.pnme.query(subject=subject, relation=relation, obj=object_val, top_k=top_k)

    def recall_associations(self, subject=None, relation=None, object_val=None, top_k=5):
        """HDC-based associative recall."""
        return self.pnme.query(subject=subject, relation=relation, obj=object_val, top_k=top_k)

    def get_context(self, keywords):
        """Get contextually relevant memories."""
        return self.pnme.get_context(keywords)

# Factory function for OpenClaw
def setup_plugin(config):
    db_path = config.get("db_path", "pnme_memory.db")
    return PNMEPlugin(db_path)
