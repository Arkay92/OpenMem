import json
from typing import List, Dict, Any
from ..api import PNME

class ClaudeMemoryAdapter:
    """
    Adapter for Claude Code tool integration.
    Wraps PNME methods into Claude-compatible tool definitions and handlers.
    """
    def __init__(self, memory_engine: PNME = None):
        self.memory = memory_engine or PNME()

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "memory_store",
                "description": "Store a specific atomic fact (Subject, Relation, Object) into long-term memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "description": "The item or concept being discussed (e.g., 'Agent X')."},
                        "relation": {"type": "string", "description": "The property or action (e.g., 'likes', 'version', 'located_at')."},
                        "object": {"type": "string", "description": "The value or target of the relationship (e.g., 'Coffee')."},
                        "context": {"type": "string", "description": "Optional context string where this was learned."},
                        "memory_type": {"type": "string", "enum": ["semantic", "episodic"], "description": "Classification of the memory."}
                    },
                    "required": ["subject", "relation", "object"]
                }
            },
            {
                "name": "memory_absorb",
                "description": "Extract and store multiple facts from a block of text automatically.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Natural language text containing facts to remember."}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "memory_query",
                "description": "Search long-term memory for facts matching a partial pattern (e.g., find all things 'Agent X' likes).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "description": "Narrow search to this subject."},
                        "relation": {"type": "string", "description": "Narrow search to this relation."},
                        "object": {"type": "string", "description": "Narrow search to this object."}
                    }
                }
            },
            {
                "name": "memory_hydrate",
                "description": "Read long-term context into the current thread for a specific topic or keyword list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "The topic or keywords to find relevant context for."}
                    },
                    "required": ["topic"]
                }
            }
        ]

    def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> str:
        try:
            if name == "memory_store":
                mid = self.memory.store(
                    arguments["subject"],
                    arguments["relation"],
                    arguments["object"],
                    source="claude_code",
                    **{k: v for k, v in arguments.items() if k not in ["subject", "relation", "object"]}
                )
                return json.dumps({"status": "success", "memory_id": mid})

            elif name == "memory_absorb":
                count = self.memory.absorb(arguments["text"], source="claude_code")
                return json.dumps({"status": "success", "facts_learned": count})

            elif name == "memory_query":
                results = self.memory.query(
                    subject=arguments.get("subject"),
                    relation=arguments.get("relation"),
                    obj=arguments.get("object")
                )
                # Format records for readable tool output
                output = []
                for res in results:
                    rec = res["record"]
                    output.append({
                        "fact": f"{rec.subject} {rec.relation} {rec.object}",
                        "score": round(res["score"], 3),
                        "explanation": res["explanation"]
                    })
                return json.dumps(output, indent=2)

            elif name == "memory_hydrate":
                context = self.memory.hydrate(arguments["topic"])
                return context

            return f"Error: Unknown tool '{name}'."
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

# Default instance for quick imports
default_adapter = ClaudeMemoryAdapter()
def get_claude_tools(): return default_adapter.get_tool_definitions()
def handle_tool_call(n, a): return default_adapter.handle_tool_call(n, a)
