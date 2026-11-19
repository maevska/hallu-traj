from __future__ import annotations

import time

import numpy as np

from .records import Record


def _version_tuple(v: str) -> tuple[int, int]:
    parts = v.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return (0, 0)


class HiddenStateExtractor:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        dtype: str = "auto",
        layers: list[int] | None = None,
        max_new_tokens: int = 64,
        max_answer_tokens: int = 128,
        state_mode: str = "producing",
        prompt_template: str = "{question}\nAnswer concisely in one or two sentences.",
        use_chat_template: bool | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if state_mode not in ("producing", "reading"):
            raise ValueError("state_mode должен быть 'producing' или 'reading'")
        self.torch = torch
        self.model_name = model_name
        self.layers = layers
        self.max_new_tokens = max_new_tokens
        self.max_answer_tokens = max_answer_tokens
        self.state_mode = state_mode
        self.prompt_template = prompt_template

        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        if use_chat_template is None:
            use_chat_template = bool(getattr(self.tok, "chat_template", None))
        self.use_chat_template = use_chat_template

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        dtypes = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        key = "dtype" if _version_tuple(transformers.__version__) >= (4, 56) else "torch_dtype"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, **{key: dtypes[dtype]}
        )
        self.model.to(device).eval()

        eos = {self.tok.eos_token_id}
        gen_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(gen_eos, int):
            eos.add(gen_eos)
        elif gen_eos:
            eos.update(gen_eos)
        self.eos_ids = {e for e in eos if e is not None}


    def _build_prompt(self, question: str):
        content = self.prompt_template.format(question=question)
        if self.use_chat_template:
            text = self.tok.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
            add_special = False  # служебные токены уже есть в шаблоне
        else:
            text = f"Q: {question}\nA:"
            add_special = True
        return self.tok(text, add_special_tokens=add_special, return_tensors="pt").input_ids.to(self.device)


    def _generate(self, question: str, sample: bool, n: int, temperature: float, top_p: float = 0.95):
        torch = self.torch
        ids = self._build_prompt(question)
        kw = dict(
            attention_mask=torch.ones_like(ids),
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=n,
            pad_token_id=self.tok.pad_token_id,
        )
        if sample:
            kw.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.inference_mode():
            out = self.model.generate(ids, **kw)
        return ids, [o[ids.shape[1]:] for o in out]

    def _strip(self, gen):
        ids = gen.tolist()
        for i, t in enumerate(ids):
            if t in self.eos_ids:
                ids = ids[:i]
                break
        return ids

    def sample(self, question: str, k: int, temperature: float = 1.0) -> list[str]:
        _, outs = self._generate(question, sample=True, n=k, temperature=temperature)
        return [self.tok.decode(self._strip(o), skip_special_tokens=True).strip() for o in outs]

    def _token_texts(self, ids: list[int]) -> list[str]:
        pieces, prev = [], ""
        for j in range(len(ids)):
            cur = self.tok.decode(ids[: j + 1], skip_special_tokens=True)
            if cur.startswith(prev):
                pieces.append(cur[len(prev):])
            else:  # неполный многобайтовый символ: запасной вариант
                pieces.append(self.tok.decode([ids[j]], skip_special_tokens=True))
            prev = cur
        return pieces

    def analyze(self, question: str, answer: str | None = None) -> Record:
        torch = self.torch
        timings: dict[str, float] = {}
        prompt_ids = self._build_prompt(question)

        t0 = time.perf_counter()
        if answer is None:
            _, outs = self._generate(question, sample=False, n=1, temperature=1.0)
            ans_list = self._strip(outs[0])
        else:
            ans_list = self.tok(answer, add_special_tokens=False).input_ids
        ans_list = ans_list[: self.max_answer_tokens]
        if not ans_list:
            raise ValueError("пустой ответ модели")
        timings["generate"] = time.perf_counter() - t0
        ans_ids = torch.tensor(ans_list, device=self.device)

        t0 = time.perf_counter()
        full = torch.cat([prompt_ids[0], ans_ids]).unsqueeze(0)
        P, T = prompt_ids.shape[1], ans_ids.shape[0]
        with torch.inference_mode():
            out = self.model(input_ids=full, output_hidden_states=True, use_cache=False)
            pos_prod = torch.arange(P - 1, P - 1 + T, device=self.device)
            pos = pos_prod if self.state_mode == "producing" else torch.arange(P, P + T, device=self.device)
            hs = torch.stack(out.hidden_states, dim=0)[:, 0]  # (L+1, P+T, d)
            hs = hs[:, pos]
            if self.layers is not None:
                hs = hs[self.layers]
            states = hs.permute(1, 0, 2).float().cpu().numpy()  # (T, L+1, d)
            logp = torch.log_softmax(out.logits[0, pos_prod].float(), dim=-1)
            lp = logp.gather(1, ans_ids[:, None])[:, 0]
            ent = -(logp.exp() * logp).sum(dim=-1)
        timings["forward"] = time.perf_counter() - t0

        text = self.tok.decode(ans_list, skip_special_tokens=True)
        return Record(
            question=question,
            answer=text.strip(),
            tokens=self._token_texts(ans_list),
            states=states.astype(np.float32),
            entropy=ent.cpu().numpy().astype(np.float64),
            logprob=lp.cpu().numpy().astype(np.float64),
            timings=timings,
        )
