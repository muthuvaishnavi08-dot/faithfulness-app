"""Streamlit demo: cross-lingual faithfulness checking for Indic RAG.

Run:  streamlit run app.py
"""
import os
import pandas as pd
import streamlit as st

from models import NLIModel, LocalLLM, pick_device
from checker import FaithfulnessChecker
from rag import load_kb, Retriever, answer_question

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
KB_DIR = os.path.join(HERE, "kb")
DEFAULT_NLI = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

st.set_page_config(page_title="Indic RAG Faithfulness Checker", page_icon="🔎", layout="wide")
st.title("🔎 Cross-lingual faithfulness checker for Indic RAG")
st.caption("Is an answer (English / Hindi / ...) supported by the retrieved English context? "
           "Detectors: numeric-consistency rule, multilingual NLI, optional LLM judge, and a fusion model.")

# ---------------------------------------------------------------- sidebar
device = pick_device()
st.sidebar.header("Settings")
st.sidebar.caption(f"Device: **{device}**")
nli_name = st.sidebar.text_input("NLI model", DEFAULT_NLI)
use_llm = st.sidebar.checkbox("Load the LLM (needed for the judge and for the RAG tab)", value=False)
llm_name = st.sidebar.selectbox("LLM", ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"], disabled=not use_llm)
four_bit = st.sidebar.checkbox("Load LLM in 4-bit (GPU + bitsandbytes)", value=False, disabled=not use_llm)
use_judge = st.sidebar.checkbox("Include the LLM judge in the score", value=False, disabled=not use_llm)
threshold = st.sidebar.slider("Flag as unfaithful when p(unfaithful) ≥", 0.05, 0.95, 0.50, 0.05)
if device == "cpu" and use_llm:
    st.sidebar.warning("No GPU found: the LLM will be slow (seconds per token).")


@st.cache_resource(show_spinner="Loading NLI model…")
def get_nli(name, dev):
    return NLIModel(name, dev)


@st.cache_resource(show_spinner="Loading LLM…")
def get_llm(name, dev, q4):
    return LocalLLM(name, dev, load_4bit=q4)


@st.cache_resource(show_spinner="Preparing checker…")
def get_checker(nli_n, llm_n, dev, q4):
    nli = get_nli(nli_n, dev)
    llm = get_llm(llm_n, dev, q4) if llm_n else None
    return FaithfulnessChecker(nli, llm, MODELS_DIR)


@st.cache_resource(show_spinner="Indexing knowledge base…")
def get_retriever(kb_key, embed_name, dev):
    return Retriever(load_kb(list(kb_key)), embed_name, dev)


def show_result(res):
    if res["verdict"].startswith("LIKELY"):
        st.error(f"**{res['verdict']}**  (p = {res['p_unfaithful']:.2f})")
    else:
        st.success(f"**{res['verdict']}**  (p(unfaithful) = {res['p_unfaithful']:.2f})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Numeric support", f"{res['rule_support']:.2f}", help="Share of numbers in the answer that also occur in the context (digits of any script are normalised).")
    c2.metric("NLI entailment", f"{res['nli_entailment']:.2f}", help="P(context entails answer) from the multilingual NLI model.")
    c3.metric("LLM judge (log-odds)", "off" if res["judge_logodds"] is None else f"{res['judge_logodds']:.2f}")
    c4.metric("Answer script", res["answer_language"])
    st.caption(f"Scoring mode: {res['scoring_mode']}")
    if res["unsupported_numbers"]:
        st.warning("Numbers in the answer that do **not** appear in the context: " + ", ".join(res["unsupported_numbers"]))
    st.markdown("**Sentence-level NLI** (which part of the answer is not supported?)")
    st.dataframe(pd.DataFrame(res["sentences"]), use_container_width=True, hide_index=True)


with st.expander("How reliable is this? (read before presenting results)"):
    st.markdown(
        "- In our experiments (English context, answers in English and Hindi), **NLI was the most reliable single detector**.\n"
        "- The **numeric rule** only catches changed numbers; it is blind to swapped names.\n"
        "- The **LLM judge** (1.5B) was weaker on entities and **unreliable on Hindi answers**; treat it as experimental.\n"
        "- The Hindi evaluation set only contained number errors, and the Hindi labels may be noisy.\n"
        "- Without `models/fusion_*.joblib` the app uses an **uncalibrated heuristic**; run `train_fusion.py` first.")

tab_check, tab_rag = st.tabs(["✅ Check an answer", "🤖 Full pipeline (retrieve → answer → check)"])

# ---------------------------------------------------------------- tab 1
PRESETS = {
    "English – faithful": ("name: Dr. Faculty 4 | department: Computer Science | subjects: Python | email: faculty4@college.edu",
                           "Dr. Faculty 4 from the Computer Science department teaches Python."),
    "English – number error": ("policy: Minimum attendance is 75% to appear in exams",
                               "The minimum attendance required is 80% to appear in exams."),
    "English – wrong entity": ("name: Dr. Faculty 4 | department: Computer Science | subjects: Python | email: faculty4@college.edu",
                               "Dr. Faculty 9 from the Computer Science department teaches Python."),
    "Hindi – faithful": ("policy: Minimum attendance is 75% to appear in exams",
                         "परीक्षा में बैठने के लिए न्यूनतम उपस्थिति ७५% होनी चाहिए।"),
    "Hindi – number error": ("policy: Minimum attendance is 75% to appear in exams",
                             "परीक्षा में बैठने के लिए न्यूनतम उपस्थिति ८०% होनी चाहिए।"),
    "Hindi – faithful but names transliterated": ("name: Dr. Faculty 4 | department: Computer Science | subjects: Python | email: faculty4@college.edu",
                                                  "डॉ. फैकल्टी 4 कंप्यूटर साइंस विभाग से हैं और पायथन पढ़ाते हैं।"),
    "Custom (type your own)": ("", ""),
}

with tab_check:
    choice = st.selectbox("Example", list(PRESETS))
    ctx0, ans0 = PRESETS[choice]
    left, right = st.columns(2)
    context = left.text_area("Retrieved context (English)", ctx0, height=170, key=f"ctx_{choice}")
    answer = right.text_area("Answer to verify (any language)", ans0, height=170, key=f"ans_{choice}")
    if st.button("Check faithfulness", type="primary"):
        if not context.strip() or not answer.strip():
            st.warning("Please provide both a context and an answer.")
        else:
            checker = get_checker(nli_name, llm_name if use_llm else None, device, four_bit)
            with st.spinner("Scoring…"):
                res = checker.check(context, answer, threshold=threshold, use_judge=use_judge)
            show_result(res)

# ---------------------------------------------------------------- tab 2
with tab_rag:
    st.markdown("Upload the knowledge-base JSON files (one per category), or put them in a `kb/` folder next to `app.py`.")
    uploads = st.file_uploader("Knowledge-base JSON files", type="json", accept_multiple_files=True)
    files = [(u.name, u.getvalue()) for u in uploads]
    if not files and os.path.isdir(KB_DIR):
        files = [(f, open(os.path.join(KB_DIR, f), "rb").read()) for f in sorted(os.listdir(KB_DIR)) if f.endswith(".json")]
    st.caption(f"{len(files)} knowledge-base file(s) available.")

    q1, q2, q3 = st.columns([4, 1, 1])
    question = q1.text_input("Question", "What is the minimum attendance requirement?")
    language = q2.selectbox("Answer in", ["English", "Hindi"])
    top_k = q3.slider("Docs in context", 1, 5, 3)

    if st.button("Retrieve, answer and check", type="primary"):
        if not files:
            st.warning("Add the knowledge-base JSON files first.")
        elif not use_llm:
            st.warning("Tick 'Load the LLM' in the sidebar: generation needs it.")
        else:
            retriever = get_retriever(tuple(files), "intfloat/multilingual-e5-base", device)
            hits = retriever.search(question, top_k=top_k)
            context = "\n".join(h["content"] for h in hits)
            st.markdown("**Retrieved documents**")
            st.dataframe(pd.DataFrame(hits)[["document_id", "category", "hybrid", "content"]],
                         use_container_width=True, hide_index=True)
            llm = get_llm(llm_name, device, four_bit)
            with st.spinner("Generating the answer…"):
                answer = answer_question(llm, context, question, language)
            st.markdown("**Generated answer**")
            st.info(answer)
            checker = get_checker(nli_name, llm_name, device, four_bit)
            with st.spinner("Checking faithfulness…"):
                res = checker.check(context, answer, threshold=threshold, use_judge=use_judge)
            show_result(res)
            st.caption("The evaluation in the notebook used a single gold document as context; multi-document contexts are an extension.")
