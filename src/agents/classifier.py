from src.utils.text_processing import load_kb, build_symptoms_vocab, extract_symptoms_from_text, normalize_text

class SymptomClassifier:
    def __init__(self, kb_path: str = None):
        self.conditions = load_kb(kb_path)
        self.symptom_vocab = build_symptoms_vocab(self.conditions)
        # build mapping symptom -> categories (may be multiple)
        self.symptoms_to_categories = {}
        for c in self.conditions:
            cat = c.get("category", "Unknown")
            for s in c.get("symptoms", []):
                key = s.lower().strip()
                self.symptoms_to_categories.setdefault(key, set()).add(cat)

    def classify(self, text: str):
        text = normalize_text(text)
        matched = extract_symptoms_from_text(text, self.symptom_vocab)
        # map to categories and counts
        cat_counts = {}
        for s in matched:
            cats = self.symptoms_to_categories.get(s, {"Unknown"})
            for cat in cats:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        
        # sort categories by count
        sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])
        return {
            "input_text": text,
            "matched_symptoms": matched,
            "categories": sorted_cats
        }
    
# quick test (run in python shell):
# clf = SymptomClassifier(); print(clf.classify("I have a sore throat and a runny nose and fever"))