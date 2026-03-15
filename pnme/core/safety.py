import re
from typing import List, Dict, Any

class SafetyFilter:
    """
    Detects and redacts sensitive information (secrets, API keys, etc.) 
    before it reaches the persistence layer.
    """
    def __init__(self):
        # Common secret patterns
        self.secret_patterns = {
            "api_key": r'(?:key|token|auth|secret|password|passwd|pwd)[\s:=]+[\'"]?([a-zA-Z0-9_\-\.]{12,})[\'"]?',
            "raw_api_key": r'\b((?:sk|pk|ak|sec|key|token)[a-zA-Z0-9\-_]{12,})\b',
            "env_secret": r'export\s+\w*SECRET=[\'"]?([a-zA-Z0-9_\-\.]{12,})[\'"]?',
            "bearer": r'Bearer\s+([a-zA-Z0-9_\-\.]{20,})',
            "possible_secret": r'\b([a-fA-F0-9]{32,}|[a-zA-Z0-9]{40,})\b'
        }

    def redact(self, text: str) -> str:
        """Redacts secrets in a string and returns the cleaned text."""
        if not isinstance(text, str):
            return text
            
        redacted_text = text
        for name, pattern in self.secret_patterns.items():
            matches = re.finditer(pattern, redacted_text, re.IGNORECASE)
            for match in matches:
                # The secret is usually in the first group if defined, otherwise the whole match
                secret = match.group(1) if match.groups() else match.group(0)
                redacted_text = redacted_text.replace(secret, f"[REDACTED_{name.upper()}]")
        
        return redacted_text

    def scrub_record(self, subject: str, relation: str, obj: str, context: str) -> Dict[str, str]:
        """Scrub all parts of a potential memory record."""
        return {
            "subject": self.redact(subject),
            "relation": self.redact(relation),
            "object": self.redact(obj),
            "context": self.redact(context)
        }
