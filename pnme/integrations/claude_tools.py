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
                    "subject": {"type": "string", "description": "The subject of the memory (e.g., 'user')"},
                    "relation": {"type": "string", "description": "The relationship (e.g., 'prefers_language')"},
                    "object": {"type": "string", "description": "The object or value (e.g., 'rust')"},
                    "context": {"type": "string", "description": "Additional context or notes"}
                },
                "required": ["subject", "relation", "object"]
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
        memory.store(arguments["subject"], arguments["relation"], arguments["object"], arguments.get("context", ""))
        return "Memory stored successfully."
    elif name == "memory_query":
        results = memory.query(subject=arguments.get("subject"), relation=arguments.get("relation"), obj=arguments.get("object"))
        return str(results)
    elif name == "memory_context":
        results = memory.retrieve_context(arguments["keywords"])
        return str(results)
    return "Unknown tool."
