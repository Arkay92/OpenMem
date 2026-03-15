from ..api import PNME

memory = PNME()

def get_claude_tools():
    return [
        {
            "name": "memory_store",
            "description": "Store a semantic memory or fact for long-term recall.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "The subject of the fact."},
                    "relation": {"type": "string", "description": "The relationship (e.g., likes, owner_of, version)."},
                    "object": {"type": "string", "description": "The object or value."},
                    "context": {"type": "string", "description": "Optional context where this was learned."},
                    "memory_type": {"type": "string", "enum": ["semantic", "episodic"], "description": "Type of memory."}
                },
                "required": ["subject", "relation", "object"]
            }
        },
        {
            "name": "memory_hydrate",
            "description": "Retrieve relevant long-term context for a given prompt or topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The prompt or text to find context for."}
                },
                "required": ["text"]
            }
        },
        {
            "name": "memory_query",
            "description": "Query long-term memory for specific facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "relation": {"type": "string"},
                    "object": {"type": "string"}
                }
            }
        },
        {
            "name": "memory_context",
            "description": "Retrieve relevant long-term memories based on current context keywords.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords to find similar memories"
                    }
                },
                "required": ["keywords"]
            }
        }
    ]

def handle_tool_call(name, arguments):
    if name == "memory_store":
        return memory.store(
            arguments["subject"],
            arguments["relation"],
            arguments["object"],
            source="claude_code",
            **{k: v for k, v in arguments.items() if k not in ["subject", "relation", "object"]}
        )
    elif name == "memory_hydrate":
        return memory.hydrator.hydrate_context(arguments["text"])
    elif name == "memory_query":
        results = memory.query(subject=arguments.get("subject"), relation=arguments.get("relation"), obj=arguments.get("object"))
        return str(results)
    elif name == "memory_context":
        results = memory.retrieve_context(arguments["keywords"])
        return str(results)
    return "Unknown tool."
