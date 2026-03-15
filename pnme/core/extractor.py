import re
from typing import List, Tuple, Dict, Any

class BaseExtractor:
    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        raise NotImplementedError

class RegexExtractor(BaseExtractor):
    """
    Rule-based fact extraction using improved regex patterns.
    """
    def __init__(self):
        # Improved patterns to handle multi-word subjects/objects
        self.patterns = [
            (r'(?P<subject>[\w\s]+) likes (?P<object>[\w\s]+)', 'likes'),
            (r'(?P<subject>[\w\s]+) is (?P<object>[\w\s]+)', 'is_a'),
            (r'(?P<subject>[\w\s]+) prefers (?P<object>[\w\s]+)', 'prefers'),
            (r'(?P<subject>[\w\s]+) works with (?P<object>[\w\s]+)', 'collaborates_with'),
            (r'(?P<subject>[\w\s]+) knows (?P<object>[\w\s]+)', 'knows'),
        ]

    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        triples = []
        for pattern, rel in self.patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for m in matches:
                subject = m.group('subject').strip().lower()
                obj = m.group('object').strip().lower()
                # Simple heuristic to avoid matching too much noise
                if len(subject) < 50 and len(obj) < 50:
                    triples.append((subject, rel, obj))
        
        return list(set(triples))

class CompositeExtractor(BaseExtractor):
    """
    Orchestrates multiple extractors and merges results.
    """
    def __init__(self, extractors: List[BaseExtractor] = None):
        self.extractors = extractors or [RegexExtractor()]

    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        all_triples = []
        for extractor in self.extractors:
            all_triples.extend(extractor.extract_triples(text))
        return list(set(all_triples))

class MemoryExtractor(CompositeExtractor):
    """Backward compatible class name for the engine."""
    def __init__(self):
        super().__init__([RegexExtractor()])

    def extract_from_logs(self, dialogue_logs: List[Dict[str, str]]) -> List[Tuple[str, str, str]]:
        all_triples = []
        for turn in dialogue_logs:
            text = turn.get('content', '')
            all_triples.extend(self.extract_triples(text))
        return list(set(all_triples))
