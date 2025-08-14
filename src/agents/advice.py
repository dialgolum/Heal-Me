from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os

LLM_MODEL = "google/flan-t5-small"  # optional local open-source T5 variant

def _template_advice(symptoms: List[str], top_conditions: List[dict]):
    """
    Build safe, non-diagnostic advice using templates and KB advice snippets.
    """
    intro = "Disclaimer: I am not a doctor. This is NOT medical advice. Possible matches (ranked):\n\n"
    lines = []
    for c in top_conditions:
        lines.append(f"- {c['name']} (score: {c['score']:.2f}, risk: {c.get('risk_level','unknown')}): {c.get('advice_snippet','')}")
    footer = ("\n\nIf you experience severe symptoms (e.g., difficulty breathing, chest pain, loss of consciousness, "
              "high persistent fever), seek emergency care immediately. For persistent or worrying symptoms, consult a healthcare professional.")
    return intro + "\n".join(lines) + footer

class AdviceAgent:
    def __init__(self, use_llm: bool = False, model_name: str = LLM_MODEL):
        self.use_llm = use_llm
        self.model_name = model_name
        self.generator = None
        if use_llm:
            # initialize a small seq2seq generator for friendly text (optional)
            try:
                self.generator = pipeline("text2text-generation", model=self.model_name, max_length=200)
            except Exception as e:
                # fallback to templates if model load fails
                print("Could not load LLM pipeline (falling back to template):", e)
                self.generator = None
                self.use_llm = False

    def generate(self, symptoms: List[str], top_conditions: List[dict]) -> str:
        if not top_conditions:
            return "No conditions matched. Please provide more details or consult a healthcare professional."
        if self.use_llm and self.generator:
            # craft a short prompt
            cond_text = "; ".join([f"{c['name']} (risk {c.get('risk_level')})" for c in top_conditions])
            prompt = (f"User symptoms: {', '.join(symptoms)}.\nPossible conditions: {cond_text}.\n"
                      "Write a concise, non-diagnostic explanation of what might be happening and safe next steps the user can take. "
                      "Include a clear disclaimer 'not medical advice'. Keep it short (4-6 sentences).")
            try:
                out = self.generator(prompt)
                return out[0]["generated_text"]
            except Exception:
                return _template_advice(symptoms, top_conditions)
        else:
            return _template_advice(symptoms, top_conditions)