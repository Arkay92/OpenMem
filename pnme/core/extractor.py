import re
import json
from typing import List, Tuple, Dict, Any, Optional

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

class LLMExtractor(BaseExtractor):
    """
    Extracts high-fidelity facts using an LLM (Anthropic Claude).
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model
        self.client = None
        if api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                pass

    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        if not self.client:
            return []

        prompt = (
            "Extract semantic facts from the following text as a JSON list of triples. "
            "Return ONLY the JSON list. Each triple should be [subject, relation, object]. "
            "Normalize to lower case. Be concise.\n\n"
            f"Text: \"{text}\""
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
            # Attempt to find JSON list in output
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return [tuple(map(str, t)) for t in data if len(t) == 3]
        except Exception:
            return []
        return []

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
    """
    The main extraction interface for PNME.
    Defaults to regex; enables LLM if api_key is provided.
    """
    def __init__(self, anthropic_key: Optional[str] = None):
        extractors = [RegexExtractor()]
        if anthropic_key:
            extractors.append(LLMExtractor(api_key=anthropic_key))
        super().__init__(extractors)

    def extract_from_logs(self, dialogue_logs: List[Dict[str, str]]) -> List[Tuple[str, str, str]]:
        all_triples = []
        for turn in dialogue_logs:
            text = turn.get('content', '')
            all_triples.extend(self.extract_triples(text))
        return list(set(all_triples))
