import re
from typing import List, Dict, Any

class ContextHydrator:
    def __init__(self, pnme, max_tokens=1000):
        self.pnme = pnme
        self.max_tokens = max_tokens  # Simplified token count (using chars/4 heuristic)

    def extract_keywords(self, text: str) -> List[str]:
        """Extract high-value keywords for memory lookup."""
        # words > 3 chars, ignoring some common stop words
        words = re.findall(r'\b\w{4,}\b', text.lower())
        stop_words = {'about', 'which', 'their', 'there', 'would', 'could', 'should', 'please', 'thank'}
        return list(set([w for w in words if w not in stop_words]))

    def hydrate_context(self, prompt: str, top_k=5) -> str:
        """
        Inject relevant long-term context into the prompt while respecting budgets.
        """
        keywords = self.extract_keywords(prompt)
        if not keywords:
            return prompt
            
        # Call engine context retrieval
        memories = self.pnme.get_context(keywords, top_k=top_k)
        if not memories:
            return prompt
            
        context_lines = []
        current_budget = self.max_tokens * 4 # Convert token budget to char estimate
        
        header = "\n<long_term_memory>\n"
        footer = "</long_term_memory>\n"
        budget_used = len(header) + len(footer)
        
        for res in memories:
            rec = res['record']
            line = f"- {rec.subject} {rec.relation} {rec.object}\n"
            if budget_used + len(line) > current_budget:
                break
            context_lines.append(line)
            budget_used += len(line)
            
        if not context_lines:
            return prompt
            
        context_block = header + "".join(context_lines) + footer
        return context_block + prompt

    def hydrate_with_template(self, prompt: str, template: str = "{context}\n{prompt}", top_k=5) -> str:
        """
        Apply context using a custom template.
        """
        keywords = self.extract_keywords(prompt)
        memories = self.pnme.get_context(keywords, top_k=top_k)
        
        context_str = ""
        if memories:
            context_str = "\n".join([f"- {m['record'].subject} {m['record'].relation} {m['record'].object}" for m in memories])
            
        return template.format(context=context_str, prompt=prompt)
