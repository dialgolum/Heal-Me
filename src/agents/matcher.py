import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.text_processing import load_kb

EMBEDDINGS_PATH = Path(__file__).resolve().parents[2] / "data" / "kb_embeddings.npz"
MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, open-source

class ConditionMatcher:
    def __init__(self, kb_path: str = None, model_name: str = MODEL_NAME):
        self.conditions = load_kb(kb_path)
        self.model = SentenceTransformer(model_name)
        # Prepare textual docs for each condition
        self.docs = [self._cond_to_doc(c) for c in self.conditions]
        # Try load embeddings if exists; otherwise compute
        if EMBEDDINGS_PATH.exists():
            arr = np.load(EMBEDDINGS_PATH)
            self.embeddings = arr["embeddings"]
        else:
            self.embeddings = self.model.encode(self.docs, convert_to_numpy=True, show_progress_bar=True)
            # don't fail if saving is not possible; try to save for future runs
            try:
                np.savez_compressed(EMBEDDINGS_PATH, embeddings=self.embeddings)
            except Exception:
                pass

    def _cond_to_doc(self, cond):
        # Create a short textual representation for IR
        name = cond.get("name", "")
        symptoms = ", ".join(cond.get("symptoms", []))
        return f"{name}. Symptoms: {symptoms}"

    def match(self, symptoms_list, top_k=5):
        """
        symptoms_list: list of symptom strings (already normalized)
        returns list of top_k matched conditions with similarity scores (0-1)
        """
        if not symptoms_list:
            return []
        query = ", ".join(symptoms_list)
        q_emb = self.model.encode([query], convert_to_numpy=True)[0]
        sims = cosine_similarity([q_emb], self.embeddings)[0]  # shape (n_conditions,)
        idxs = np.argsort(-sims)[:top_k]
        results = []
        for i in idxs:
            cond = self.conditions[int(i)]
            results.append({
                "id": cond.get("id"),
                "name": cond.get("name"),
                "score": float(sims[int(i)]),
                "category": cond.get("category"),
                "risk_level": cond.get("risk_level"),
                "advice_snippet": cond.get("advice", "")[:300]
            })
        return results

# Example usage:
# m = ConditionMatcher(); print(m.match(["sore throat","fever"]))