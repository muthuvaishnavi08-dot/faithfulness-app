# Cross-lingual faithfulness checker (Streamlit demo)

Checks whether an answer (English / Hindi / ...) is supported by an **English** retrieved context, using
a numeric-consistency rule, a multilingual NLI model, an optional LLM judge, and a small fusion model.
Optionally runs the whole pipeline: retrieve from your knowledge base → generate an answer → check it.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (two tabs) |
| `checker.py` | Numeric rule, sentence-level NLI, fusion scoring |
| `models.py` | NLI model + local LLM wrappers |
| `rag.py` | KB loading, hybrid retriever, answer generation |
| `train_fusion.py` | Trains the fusion model from the notebook's `dataset_with_scores.csv` |

## Setup
```bash
pip install -r requirements.txt
```

1. **(Recommended) fusion model.** Download `dataset_with_scores.csv` from the notebook's Kaggle output, then:
   ```bash
   python train_fusion.py dataset_with_scores.csv      # creates models/fusion_*.joblib
   ```
   Without it the app falls back to an *uncalibrated heuristic*.
2. **Knowledge base (only for the RAG tab).** Put the JSON files in a `kb/` folder next to `app.py`, or upload them in the app.
3. **Run**
   ```bash
   streamlit run app.py
   ```

## Notes
* First run downloads the models (~0.6 GB NLI, ~3 GB for Qwen2.5-1.5B, ~1 GB for the e5 embedder).
* **Tab 1 works with NLI only** (CPU is fine). The LLM judge and the RAG tab need the LLM; use a GPU.
* On Colab/Kaggle, run `streamlit run app.py &` and expose it with a tunnel (e.g. `cloudflared`, `ngrok`).
  Streamlit Community Cloud has too little memory for the LLM; NLI-only mode may fit.
* Reliability caveats (judge unreliable on Hindi, Hindi labels possibly noisy) are summarised inside the app.
