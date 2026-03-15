import re
from typing import List, Dict, Any

class ContextHydrator:
    def __init__(self, pnme, max_tokens=800):
        self.pnme = pnme
        self.max_tokens = max_tokens
        # 3 chars per token is a more conservative estimate for structured triples
        self.chars_per_token = 3.0

    def extract_keywords(self, text: str) -> List[str]:
        """Extract high-value keywords for memory lookup."""
        # words > 3 chars, ignoring some common stop words
        words = re.findall(r'\b\w{4,}\b', text.lower())
        stop_words = {
            'about', 'which', 'their', 'there', 'would', 'could', 'should', 
            'please', 'thank', 'where', 'these', 'those', 'using', 'through'
        }
        return list(set([w for w in words if w not in stop_words]))

    def hydrate_context(self, prompt: str, top_k=8, budget_override: int = None) -> str:
        """
        Inject relevant long-term context into the prompt while respecting token budgets.
        """
        keywords = self.extract_keywords(prompt)
        if not keywords:
            return prompt
            
        # Call engine context retrieval (Stage 15 query_text integration)
        memories = self.pnme.get_context(keywords, top_k=top_k)
        if not memories:
            return prompt
            
        context_lines = []
        max_budget = budget_override if budget_override else self.max_tokens
        char_budget = int(max_budget * self.chars_per_token)
        
        header = "\n[LONG-TERM MEMORY CONTEXT]\n"
        footer = "\n"
        budget_used = len(header) + len(footer)
        
        # Deduplication and relevance filtering
        seen_ids = set()
        for res in memories:
            rec = res['record']
            if rec.memory_id in seen_ids:
                continue
            
            # Formatting for LLM digest
            line = f"• {rec.subject} {rec.relation} {rec.object}"
            if rec.context:
                line += f" (Context: {rec.context})"
            line += "\n"
            
            if budget_used + len(line) > char_budget:
                break
                
            context_lines.append(line)
            budget_used += len(line)
            seen_ids.add(rec.memory_id)
            
        if not context_lines:
            return prompt
            
        context_block = header + "".join(context_lines) + footer
        return context_block + prompt

    def hydrate_with_template(self, prompt: str, template: str = "{context}\n{prompt}", top_k=5) -> str:
        """
        Apply context using a custom template.
        """
        context_str = self.hydrate_context(prompt, top_k=top_k)
        if context_str == prompt:
            return prompt
        
        # Extract the context block only
        if "[LONG-TERM MEMORY CONTEXT]" in context_str:
            parts = context_str.split("[LONG-TERM MEMORY CONTEXT]")
            # Assuming it's at the start
            context_content = parts[1].split("\n\n")[0] # Simple split heuristic
            return template.format(context=f"[MEMORY]\n{context_content}", prompt=prompt)
        
        return context_str
