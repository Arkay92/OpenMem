import re
from typing import List, Dict, Any

class ContextHydrator:
    def __init__(self, pnme):
        self.pnme = pnme

    def extract_keywords(self, text: str) -> List[str]:
        """Extract potential keywords for memory lookup."""
        # Simple extraction: words > 4 chars, ignoring common stop words
        words = re.findall(r'\b\w{4,}\b', text.lower())
        stop_words = {'about', 'which', 'their', 'there', 'would', 'could', 'should'}
        return [w for w in words if w not in stop_words]

    def hydrate_context(self, prompt: str, top_k=3) -> str:
        """
        Search memory for prompt keywords and prepend relevant facts to the prompt.
        """
        keywords = self.extract_keywords(prompt)
        if not keywords:
            return prompt
            
        memories = self.pnme.retrieve_context(keywords)
        if not memories:
            return prompt
            
        context_str = "\n[Relevant Long-term Memories]:\n"
        for res in memories[:top_k]:
            rec = res['record']
            context_str += f"- {rec.subject} {rec.relation} {rec.object} (Source: {rec.source})\n"
        
        return context_str + "\n" + prompt
