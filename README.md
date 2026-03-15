# OpenMem: Persistent Neuro-Symbolic Memory (PNME)

**A high-performance, persistent neuro-symbolic memory engine for LLM agents using Hyperdimensional Computing (HDC) and symbolic triples.**

OpenMem (PNME) provides a durable long-term memory layer that survives across sessions, tools, and agents. It acts as a memory coprocessor, allowing agents to store facts, retrieve relevant context, and learn associations over time.

## Installation

```bash
# Clone the repository
git clone https://github.com/EasyTees/OpenMem
cd OpenMem

# Install as a package
pip install -e .
```

## Core Features

- **Principled HDC Encoding**: Uses 10,000-dimensional bipolar vectors with deterministic hash-based seeds and role-vector binding.
- **Hybrid Retrieval**: Combines exact symbolic filtering with sub-symbolic HDC unbinding and multi-factor ranking (recency, strength, provenance).
- **Audit & Analytics**: Integrated access logging and event tracking for memory lifecycle analysis.
- **Portability**: JSONL export/import support for cross-system memory migration.
- **Safety**: Automated scrubbing of sensitive information (secrets, keys) before storage.

## Agent Integrations

### 1. Claude Code

OpenMem integrates with Claude via a structured tool adapter. To use it, import the adapter and register the tools:

```python
from pnme.integrations.claude_tools import ClaudeMemoryAdapter

# Initialize adapter
adapter = ClaudeMemoryAdapter(db_path="my_memory.db")
tools = adapter.get_tool_definitions()

# In your Claude tool handler:
def on_tool_call(name, args):
    return adapter.handle_tool_call(name, args)
```

**Available Tools:**
- `memory_store`: Save a specific fact (S, R, O).
- `memory_absorb`: Bulk-extract facts from a text block.
- `memory_query`: Pattern-based retrieval.
- `memory_hydrate`: Inject context into the prompt.

### 2. OpenClaw

Register OpenMem as a plugin in your OpenClaw setup:

```python
from pnme.integrations.openclaw_plugin import setup_plugin

# Setup the memory plugin
memory_plugin = setup_plugin({"db_path": "openclaw_mem.db"})
agent.register_plugin(memory_plugin)
```

## Quick Start (Python)

```python
from pnme.api import PNME

memory = PNME()

# Store a fact
memory.store("DeepSeek-V3", "released_by", "DeepSeek")

# Absorb facts from text
memory.absorb("Claude 3.5 Sonnet is a model by Anthropic. It supports computer use.")

# Retrieve hydrated context
prompt = "Tell me about Anthropic models."
hydrated_prompt = memory.hydrate(prompt)
```

## License

MIT
