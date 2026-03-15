import re
from typing import List, Tuple, Dict

class MemoryExtractor:
    """
    Extracts semantic facts (Subject, Relation, Object) from text.
    In a production system, this would involve an LLM call.
    For this version, we provide a rule-based stub and a structure for LLM integration.
    """
    def __init__(self):
        # Basic patterns for rule-based fallback
        self.patterns = [
            (r'(?P<subject>\w+) likes (?P<object>\w+)', 'likes'),
            (r'(?P<subject>\w+) is (?P<object>[\w\s]+)', 'is_a'),
            (r'(?P<subject>\w+) prefers (?P<object>\w+)', 'prefers'),
            (r'(?P<subject>\w+) works with (?P<object>\w+)', 'collaborates_with')
        ]

    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Distill triples from text.
        Returns a list of (Subject, Relation, Object).
        """
        triples = []
        for pattern, rel in self.patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for m in matches:
                triples.append((m.group('subject').strip().lower(), rel, m.group('object').strip().lower()))
        
        return list(set(triples)) # Deduplicate

    def extract_from_logs(self, dialogue_logs: List[Dict[str, str]]) -> List[Tuple[str, str, str]]:
        """
        Process a list of dialogue turns (role/content) to find facts.
        """
        all_triples = []
        for turn in dialogue_logs:
            text = turn.get('content', '')
            all_triples.extend(self.extract_triples(text))
        
        return list(set(all_triples))
