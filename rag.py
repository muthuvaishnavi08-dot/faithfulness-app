"""Minimal RAG pieces for the demo: KB loading, hybrid retriever, answer generation."""
import json, re
import numpy as np

CATEGORY_DESC = {
    "faculty": "Faculty information. Professor, teacher, instructor, staff member and the subjects they teach.",
    "courses": "Academic course information. Course name, semester, credits and associated faculty.",
    "attendance_policy": "Student attendance rules, minimum attendance percentage, attendance requirements and shortage policies.",
    "campus_services": "University campus services, facilities, opening hours, service timings and student facilities.",
    "faq": "Frequently asked questions and answers about university services, student issues and administrative procedures.",
    "exam_schedule": "University examination timetable, subject, exam date and examination time.",
    "academic_calendar": "Academic calendar, semester dates, university events, registration and academic activities.",
    "holidays": "University holidays, public holidays and holiday dates.",
    "hostel": "Hostel rules, accommodation regulations, visitor timings and hostel facilities.",
    "placement_policy": "Campus placement eligibility, recruitment rules, CGPA requirements and placement policies.",
}

GEN_SYSTEM = ("You are a university information assistant. Answer the question using ONLY the context. "
              "Be brief (one or two sentences). Copy names, numbers, dates, times and email addresses exactly as written in the context. "
              "Never translate or transliterate names: keep them in Latin letters exactly as in the context.")


def clean(t):
    return re.sub(r"\s+", " ", str(t)).strip()


def as_records(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        inner = next((v for v in obj.values() if isinstance(v, list)), None)
        return inner if inner is not None else [obj]
    return []


def load_kb(files):
    """files: list of (filename, bytes|str). One JSON file per category (file stem = category)."""
    docs = []
    for fname, raw in files:
        category = fname.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        obj = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
        for rec in as_records(obj):
            if not isinstance(rec, dict) or not rec.get("document_id"):
                continue
            fields = {k: (", ".join(map(str, v)) if isinstance(v, list) else str(v))
                      for k, v in rec.items() if k != "document_id" and v is not None}
            content = clean(" | ".join(f"{k}: {v}" for k, v in fields.items()))
            if content:
                docs.append({"document_id": rec["document_id"], "category": category, "content": content,
                             "passage": f"{CATEGORY_DESC.get(category, category.replace('_', ' '))} Details: {content}"})
    return docs


def _minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class Retriever:
    def __init__(self, docs, embed_name="intfloat/multilingual-e5-base", device="cpu"):
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.docs = docs
        self.is_e5 = "e5" in embed_name.lower()
        self.embedder = SentenceTransformer(embed_name, device=device)
        self.doc_emb = self.embedder.encode([self._p(d["passage"]) for d in docs], batch_size=32,
                                            normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        self.tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
        self.tfidf_mat = self.tfidf.fit_transform([d["passage"] for d in docs])

    def _q(self, q): return ("query: " + q) if self.is_e5 else q
    def _p(self, p): return ("passage: " + p) if self.is_e5 else p

    def search(self, query, top_k=3, alpha=0.8):
        qe = self.embedder.encode([self._q(query)], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        sem = (qe @ self.doc_emb.T)[0]
        kw = (self.tfidf.transform([query]) @ self.tfidf_mat.T).toarray()[0]
        score = alpha * _minmax(sem) + (1 - alpha) * _minmax(kw)
        order = np.argsort(-score)[:top_k]
        return [{**self.docs[i], "hybrid": float(score[i]), "cosine": float(sem[i])} for i in order]


def answer_question(llm, context, question, language):
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer in {language}."
    return llm.chat(GEN_SYSTEM, user, max_new_tokens=120)
