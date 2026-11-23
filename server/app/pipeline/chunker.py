import re
from typing import List

# Common abbreviations that should NOT trigger a sentence boundary when followed by a period
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
    "vs", "etc", "eg", "e.g", "ie", "i.e",
    "approx", "appx", "apt", "dept", "est", "fig", "inc", "corp", "co", "ltd",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "am", "pm", "st", "ave", "rd", "blvd"
}

# Regex to detect sentence terminal punctuation: . ? ! \n
TERMINAL_PUNCT_RE = re.compile(r'([.?!]+(?:\s+|$)|[\n\r]+)')

class SentenceChunker:
    """Buffers streamed LLM tokens and yields complete spoken sentences.
    
    Ensures abbreviations, numbers, and ellipsis are handled safely without premature splits.
    """

    def __init__(self):
        self.buffer = ""

    def reset(self):
        self.buffer = ""

    def _is_abbreviation(self, text_before_punct: str) -> bool:
        """Check if the word immediately preceding the period is a known abbreviation."""
        tokens = text_before_punct.strip().split()
        if not tokens:
            return False
        last_word = tokens[-1].lower().rstrip(".").strip()
        return last_word in ABBREVIATIONS

    def _is_numeric_decimal(self, full_text: str, match_start: int) -> bool:
        """Check if period is part of a number (e.g., 3.14 or $1.50)."""
        if match_start > 0 and match_start < len(full_text) - 1:
            prev_char = full_text[match_start - 1]
            next_char = full_text[match_start + 1]
            if prev_char.isdigit() and (next_char.isdigit() or next_char in ",%"):
                return True
        return False

    def add_token(self, token: str) -> List[str]:
        """Add a token to the buffer and return any completed sentences."""
        # Sanitize token (strip XML-style think tags if any appear)
        token = re.sub(r'<think>.*?</think>', '', token, flags=re.DOTALL)
        self.buffer += token

        sentences = []
        search_start = 0

        while True:
            match = TERMINAL_PUNCT_RE.search(self.buffer, search_start)
            if not match:
                break

            punct_start = match.start()
            punct_end = match.end()
            preceding_text = self.buffer[:punct_start]
            punct = self.buffer[punct_start:punct_end]

            # Check if this period is part of a decimal number
            if "." in punct and self._is_numeric_decimal(self.buffer, punct_start):
                search_start = punct_end
                continue

            # Check if this period is an abbreviation
            if "." in punct and self._is_abbreviation(preceding_text):
                search_start = punct_end
                continue

            # Valid sentence boundary found
            complete_sentence = (preceding_text + punct.strip()).strip()
            self.buffer = self.buffer[punct_end:]
            search_start = 0

            if complete_sentence:
                sentences.append(complete_sentence)

        return sentences

    def flush(self) -> List[str]:
        """Flush any remaining buffered text as the final sentence."""
        remaining = self.buffer.strip()
        self.buffer = ""
        if remaining:
            return [remaining]
        return []
