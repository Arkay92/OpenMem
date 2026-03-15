# PNME: Persistent Neuro-Symbolic Memory Engine

PNME is a long-term memory layer for AI agents, combining **Hyperdimensional Computing (HDC)** with **Symbolic Reasoning**. It works with Claude Code, OpenClaw, and any Python agent framework.

## Installation

```bash
pip install .
```

## Features

- **Semantic Memory**: Store and retrieve facts as (Subject, Relation, Object) triples.
- **HDC Encoding**: Uses 10,000-dimensional vectors for robust associative recall.
- **Persistence**: SQLite-backed storage for cross-session long-term memory.
- **Agent Integration**: Pre-built tools for Claude Code and plugins for OpenClaw.

## Quick Start

```python
from pnme.api import PNME

# Initialize engine
memory = PNME("memory.db")

# Store a fact
memory.store("user", "prefers", "rust", context="session_1")

# Query with associative recall
results = memory.query(subject="user", relation="prefers")
print(results[0]['symbol']) # Output: rust
```

## Agent Integrations

### Claude Code

To use PNME with Claude Code, you can register the memory tools in your agent loop:

```python
from pnme.integrations.claude_tools import get_claude_tools, handle_tool_call

# Get tool definitions for Claude's tool use
tools = get_claude_tools()

# Example usage in a tool handler
# result = handle_tool_call(name="memory_store", arguments={"subject": "user", ...})
```

Supported tools: `memory_store`, `memory_query`, `memory_context`.

### OpenClaw

To use PNME as an OpenClaw plugin, initialize the plugin and register its skills:

```python
from pnme.integrations.openclaw_plugin import setup_plugin

# Initialize plugin
plugin = setup_plugin({"db_path": "pnme_memory.db"})

# Register skills
skills = plugin.get_skills()
# Skills: store_memory, query_memory, recall_associations, retrieve_context
```

## Project Structure

- `pnme/hdc`: Vector operations and encoding.
- `pnme/storage`: Persistence layer.
- `pnme/core`: Main engine and recall logic.
- `pnme/integrations`: Claude and OpenClaw adapters.

## License

MIT
