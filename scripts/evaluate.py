import json
from src.agents.classifier import SymptomClassifier
from src.agents.matcher import ConditionMatcher

def evaluate(eval_path="data/eval_dataset.json"):
    clf = SymptomClassifier()
    matcher = ConditionMatcher()

    with open(eval_path, "r", encoding="utf-8") as f:
        tests = json.loads(f)

    total = len(tests)
    symptom_hits = 0
    condition_hits = 0

    for case in tests:
        result = clf.classify(case["input_text"])
        matched_symptoms = set(result["matched_symptoms"])

        if set(case["expected_symptoms"]).issubset(matched_symptoms):
            symptom_hits += 1

        matches = matcher.match(result["matched_symptoms"], top_k=3)
        matched_conditions = {m["name"] for m in matches}
        if set(case["expected_conditions"]).intersection(matched_conditions):
            condition_hits += 1

    print(f"Symptom Extraction Accuracy: {symptom_hits}/{total} ({symptom_hits/total:.1%})")
    print(f"Condition Matching Accuracy: {condition_hits}/{total} ({condition_hits/total:.1%})")    