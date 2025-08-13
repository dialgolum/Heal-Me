from pathlib import Path
import json
import re

def load_kb(kb_path: str = None):
    """
    Load conditions KB from data/conditions.json
    Returns list of condition dicts.
    """

    if kb_path:
        p = Path(kb_path)

    else:
        p = Path(__file__).resolve().parents[2] / "data" / "conditions.json"

    with open(p, "r", encoding="utf-8") as f:
        conditions = json.loads(f)

    return conditions

def build_symptoms_vocab(conditions):
    """
    Build a sorted list of symptom phrases from the KB (longest first).
    Returns list of canonical symptom strings.
    """
    phrases = set()
    for c in conditions:
        for s in c.get("symptoms", []):
            phrases.add(s.lower().strip())

    # Sort by length descending so multi-word phrases match first
    return sorted(list(phrases), key=lambda x: -len(x))

def extract_symptoms_from_text(text: str, vocab_phrases):
    """
    Simple phrase matching: returns list of matched canonical symptoms (lowercased).
    """

    if not text or not vocab_phrases:
        return []
    
    text_1 = text.lower()
    matched = []
    for phrase in vocab_phrases:
        # woed boundary search
        if re.search(r'\b' + re.escape(phrase) + r'\b', text_1):
            matched.append(phrase)

    # remove dduplicates while preserving order
    seen = set()
    result = []
    for s in matched:
        if s not in seen:
            result.append(s)
            seen.add(s)
    
    return result

def normalize_text(text: str):
    #basec normalization
    return " ".join(text.strip().split())