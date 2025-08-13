import sys
from pathlib import Path

# Add project root to sys.path BEFORE importing from src
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
from src.agents.classifier import SymptomClassifier
from src.agents.matcher import ConditionMatcher
from src.agents.advice import AdviceAgent

st.set_page_config(page_title="Symptom Checker (Prototype)", layout="wide")

st.title("Symptom Checker — Prototype")
st.markdown(
    "Enter the symptoms you or another person has. This is a **prototype** — not medical advice. "
    "All processing is local or using free open-source models."
)

with st.sidebar:
    st.header("Settings")
    use_llm = st.checkbox("Use local LLM for richer advice (optional & may be slow)", value=False)
    top_k = st.slider("Number of matched conditions to show", 1, 10, 5)
    if st.button("Precompute KB embeddings (recommended)"):
        with st.spinner("Precomputing embeddings..."):
            import subprocess, sys
            subprocess.run([sys.executable, "scripts/prepare_kb.py"])
            st.success("Done (if script succeeded).")

# instantiate agents (lazy)
@st.cache_resource
def init_agents(use_llm_flag):
    clf = SymptomClassifier()
    matcher = ConditionMatcher()
    advice = AdviceAgent(use_llm=use_llm_flag)
    return clf, matcher, advice

clf, matcher, advice_agent = init_agents(use_llm)

st.subheader("Describe symptoms (free text)")
user_input = st.text_area("Examples: 'I have a sore throat, runny nose, and mild fever for 2 days' ", height=140)
col1, col2 = st.columns([1, 1])

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some symptoms or text.")
    else:
        with st.spinner("Classifying symptoms..."):
            classification = clf.classify(user_input)
        st.success("Done")

        with col1:
            st.markdown("### Extracted Symptoms")
            if classification["matched_symptoms"]:
                for s in classification["matched_symptoms"]:
                    st.write(f"- {s}")
            else:
                st.write("_No known symptoms matched from KB._")

            st.markdown("### Inferred Categories")
            if classification["categories"]:
                for cat, cnt in classification["categories"]:
                    st.write(f"- {cat} (matched {cnt} symptom(s))")
            else:
                st.write("_No category inferred_")

        with col2:
            st.markdown("### Condition Matching (IR)")
            matches = matcher.match(classification["matched_symptoms"], top_k=top_k)
            if matches:
                df = pd.DataFrame([{
                    "Name": m["name"],
                    "Score": round(m["score"], 3),
                    "Category": m.get("category"),
                    "Risk": m.get("risk_level")
                } for m in matches])
                st.table(df)

                # show detailed advices
                st.markdown("### Advice / Explanation")
                advice_text = advice_agent.generate(classification["matched_symptoms"], matches)
                st.write(advice_text)

                # provide JSON or text download
                out = {
                    "input": classification["input_text"],
                    "matched_symptoms": classification["matched_symptoms"],
                    "categories": classification["categories"],
                    "matches": matches,
                    "advice": advice_text
                }
                st.download_button("Download result (JSON)", data=str(out), file_name="symptom_report.json")
            else:
                st.write("_No condition matches found based on current KB._")
                st.write("Try more explicit symptoms (eg. 'severe headache', 'shortness of breath')")

st.markdown("---")
st.caption("This is a student project prototype. For production use you must: validate KB sources, implement stronger NLP (NER), add explainability logs, perform Responsible AI safety checks, and consult clinicians.")