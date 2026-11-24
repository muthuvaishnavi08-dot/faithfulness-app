"""Faithfulness checking: numeric rule + multilingual NLI (+ optional LLM judge) + optional fusion model."""
import os, re, unicodedata

NUM_RE = re.compile(r"\d+(?:\.\d+)?")

JUDGE_SYSTEM = "You are a strict fact checker."
JUDGE_USER = ("Context:\n{context}\n\nAnswer:\n{answer}\n\n"
              "Is every fact in the answer (names, numbers, dates, rules) supported by the context? Reply with Yes or No only.")

SCRIPTS = [("Hindi", 0x0900, 0x097F), ("Bengali", 0x0980, 0x09FF), ("Punjabi", 0x0A00, 0x0A7F),
           ("Gujarati", 0x0A80, 0x0AFF), ("Tamil", 0x0B80, 0x0BFF), ("Telugu", 0x0C00, 0x0C7F),
           ("Kannada", 0x0C80, 0x0CFF), ("Malayalam", 0x0D00, 0x0D7F)]


def normalize_digits(text):
    """Devanagari / Tamil / Telugu / ... digits -> ASCII so numbers can be compared across scripts."""
    return "".join(str(unicodedata.decimal(c)) if unicodedata.decimal(c, None) is not None else c for c in text)


def numbers(text):
    return set(NUM_RE.findall(normalize_digits(text)))


def split_sentences(text):
    parts = re.split(r"(?<=[.!?\u0964])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def detect_language(text):
    counts = {}
    for ch in text:
        cp = ord(ch)
        for name, lo, hi in SCRIPTS:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
    return max(counts, key=counts.get) if counts else "English/Latin script"


def numeric_report(context, answer):
    ans, ctx = numbers(answer), numbers(context)
    return {"support": 1.0 if not ans else len(ans & ctx) / len(ans),
            "answer_numbers": sorted(ans, key=float),
            "unsupported": sorted(ans - ctx, key=float)}


class FaithfulnessChecker:
    """nli / llm are duck-typed (NLIModel / LocalLLM from models.py), which keeps this class easy to test."""

    def __init__(self, nli, llm=None, models_dir="models"):
        self.nli, self.llm = nli, llm
        self.fusion = {}
        try:
            import joblib
            for name in ("fusion_full", "fusion_nojudge"):
                path = os.path.join(models_dir, f"{name}.joblib")
                if os.path.exists(path):
                    self.fusion[name] = joblib.load(path)
        except Exception:
            self.fusion = {}

    def check(self, context, answer, threshold=0.5, use_judge=True):
        num = numeric_report(context, answer)
        nli_p = float(self.nli.entail([context], [answer])[0])
        sents = split_sentences(answer) or [answer]
        sent_p = self.nli.entail([context] * len(sents), sents)

        judge = None
        if use_judge and self.llm is not None:
            judge = float(self.llm.yes_no_score(JUDGE_SYSTEM, JUDGE_USER.format(context=context, answer=answer)))

        if judge is not None and "fusion_full" in self.fusion:
            prob, mode = float(self.fusion["fusion_full"].predict_proba([[num["support"], nli_p, judge]])[0, 1]), "fusion (rule + NLI + judge)"
        elif "fusion_nojudge" in self.fusion:
            prob, mode = float(self.fusion["fusion_nojudge"].predict_proba([[num["support"], nli_p]])[0, 1]), "fusion (rule + NLI)"
        else:
            prob, mode = 1.0 - (0.5 * num["support"] + 0.5 * nli_p), "uncalibrated heuristic (no fusion model found)"

        return {"rule_support": num["support"], "unsupported_numbers": num["unsupported"],
                "answer_numbers": num["answer_numbers"], "nli_entailment": nli_p, "judge_logodds": judge,
                "p_unfaithful": prob, "scoring_mode": mode,
                "verdict": "LIKELY UNFAITHFUL" if prob >= threshold else "likely faithful",
                "sentences": [{"sentence": s, "entailment": float(p)} for s, p in zip(sents, sent_p)],
                "answer_language": detect_language(answer)}
