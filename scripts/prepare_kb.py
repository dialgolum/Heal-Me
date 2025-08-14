import sys
from pathlib import Path

# Add project root to PYTHONPATH
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

"""
Precompute condition embeddings and save to data/kb_embeddings.npz
Run this once after adding/updating data/conditions.json to speed up the app.
"""
from src.agents.matcher import ConditionMatcher
print("Computing and saving KB embeddings (this may download sentence-transformers model on first run)...")
m = ConditionMatcher()
print("Saved embeddings (if writeable). Number of conditions:", len(m.docs))