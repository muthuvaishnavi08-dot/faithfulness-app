"""Thin wrappers around the NLI model and the local LLM.
torch / transformers are imported lazily so the rest of the code can be tested without them."""


def pick_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class NLIModel:
    """Multilingual NLI. premise = retrieved context (English), hypothesis = answer (any language)."""

    def __init__(self, name, device):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch, self.device, self.name = torch, device, name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self.ent = [int(i) for i, l in self.model.config.id2label.items() if l.lower().startswith("entail")][0]

    def entail(self, premises, hyps, bs=16):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(premises), bs):
                enc = self.tok(premises[i:i + bs], hyps[i:i + bs], truncation=True, max_length=512,
                               padding=True, return_tensors="pt").to(self.device)
                p = self.torch.softmax(self.model(**enc).logits, dim=-1)[:, self.ent]
                out.extend(p.float().cpu().tolist())
        return out


class LocalLLM:
    """Instruction-tuned causal LM used for (a) answer generation and (b) the LLM-as-judge score."""

    def __init__(self, name, device, load_4bit=False):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch, self.device, self.name = torch, device, name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if load_4bit:
            from transformers import BitsAndBytesConfig
            self.model = AutoModelForCausalLM.from_pretrained(
                name, device_map="auto",
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)).eval()
        else:
            dtype = torch.float16 if device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        self.yes_id = self.tok.encode("Yes", add_special_tokens=False)[0]
        self.no_id = self.tok.encode("No", add_special_tokens=False)[0]

    def _prompt(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def chat(self, system, user, max_new_tokens=120):
        enc = self.tok(self._prompt(system, user), return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def yes_no_score(self, system, user):
        """log-odds of 'Yes' vs 'No' as the first answer token (higher = 'Yes')."""
        enc = self.tok(self._prompt(system, user), return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            logits = self.model(**enc).logits[0, -1]
        return float(logits[self.yes_id] - logits[self.no_id])
