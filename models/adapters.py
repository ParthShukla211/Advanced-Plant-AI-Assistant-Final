# models/adapters.py
# Model adapters for different model types

import hashlib
import os
from typing import List, Optional, Tuple, Dict, Any
import streamlit as st
from ctransformers import AutoModelForCausalLM
from transformers import pipeline
from huggingface_hub import hf_hub_download

# ---- ctransformers choices (local file OR HF auto-download) ----
CTRANS_CHOICES: Dict[str, Dict[str, Any]] = {
    "🟢 Local Mistral (GGUF)": {
        "path": r"C:\Advanced Plant AI Assistant\models\DeepSeek-R1-Distill-Llama-8B-Q4_0.gguf",
        "model_type": "mistral",
        "template": "mistral"
    },
    "🟡 Mistral-7B-Instruct (Q4_K_M, auto)": {
        "path": r"C:\Advanced Plant AI Assistant\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "model_type": "mistral",
        "template": "mistral"
    },
        
    "🟠 Local Phi-2 (Q5_0)": {
        "path": r"C:\Advanced Plant AI Assistant\models\phi-2.Q5_0.gguf",  # <-- Update this path if needed
        "model_type": "phi-2",
        "template": "default"  # Use "default" if you're unsure; you can tweak later
    },

    "🟣 Llama-2-7B-Chat (Q3_K_S, auto)": {
        "path": r"C:\Advanced Plant AI Assistant\models\llama-2-7b-chat.Q3_K_S.gguf",
        "model_type": "llama",
        "template": "llama2"
    },
    "🟤 Qwen2-7B-Instruct (Q4_K_M, auto)": {
        "path": r"C:\Advanced Plant AI Assistant\models\qwen2-7b-instruct-q4_k_m.gguf",
        "model_type": "qwen2",  # fallback to "qwen" if needed
        "template": "chatml"
    },
}

# ---- HF text2text choices ----
HF_T2T_CHOICES = {
    "🔵 Flan-T5 Small (HF)": "google/flan-t5-small",
    "🔵 Flan-T5 Base (HF)": "google/flan-t5-base",
    "🔵 Flan-T5 Large (HF)": "google/flan-t5-large",
}

MODEL_FAMILY = list(CTRANS_CHOICES.keys()) + list(HF_T2T_CHOICES.keys())

def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _ctrans_resolve_local_or_repo(spec: Dict[str, Any]) -> Tuple[str, str, bool]:
    if "path" in spec:
        p = spec["path"]
        if os.path.exists(p):
            base, model_file = os.path.split(p)
            return base, model_file, False
    return spec["repo_id"], spec["filename"], True

class CTransChatModel:
    def __init__(self, spec: Dict[str, Any], *, context_length: int = 4096, gpu_layers: int = 0, seed: Optional[int]=None,
                 mirostat_mode:int=0, mirostat_tau:float=5.0, mirostat_eta:float=0.1, stop_sequences: Optional[List[str]]=None):
        self.spec = spec
        self.context_length = context_length
        self.gpu_layers = gpu_layers
        self.model = None
        self.model_type = spec.get("model_type", "mistral")
        self.seed = seed
        self.mirostat_mode = mirostat_mode
        self.mirostat_tau = mirostat_tau
        self.mirostat_eta = mirostat_eta
        self.stop_sequences = stop_sequences or []

    def load(self):
        if self.model is None:
            base, model_file, is_repo = _ctrans_resolve_local_or_repo(self.spec)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    base,
                    model_file=model_file,
                    model_type=self.model_type,
                    context_length=self.context_length,
                    gpu_layers=self.gpu_layers
                )
            except Exception as e:
                if self.model_type == "qwen2":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        base, model_file=model_file, model_type="qwen",
                        context_length=self.context_length, gpu_layers=self.gpu_layers
                    )
                else:
                    raise e
        return self.model

    @staticmethod
    def _format_prompt(template: str, system_prompt: str, user_prompt: str) -> str:
        sys = (system_prompt or "You are a helpful, precise industrial assistant.").strip()
        up = (user_prompt or "").strip()
        if template in ("mistral", "llama2"):
            return f"<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{up} [/INST]"
        elif template == "chatml":  # Qwen-style
            return f"""<|im_start|>system
{sys}<|im_end|>
<|im_start|>user
{up}<|im_end|>
<|im_start|>assistant
"""
        else:
            return f"{sys}\n\n{up}\n"

    def _build_kwargs(self, max_new_tokens, temperature, top_p, top_k, repeat_penalty, stream):
        kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repeat_penalty),
            stream=bool(stream),
        )
        if self.stop_sequences:
            kwargs["stop"] = self.stop_sequences
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        if self.mirostat_mode in (1,2):
            kwargs["mirostat"] = int(self.mirostat_mode)
            kwargs["mirostat_tau"] = float(self.mirostat_tau)
            kwargs["mirostat_eta"] = float(self.mirostat_eta)
        return kwargs

    def generate(self, template: str, system_prompt: str, user_prompt: str, *,
                 max_new_tokens=512, temperature=0.3, top_p=0.9, top_k=40,
                 repeat_penalty=1.2, stream=False):
        model = self.load()
        prompt = self._format_prompt(template, system_prompt, user_prompt)
        kwargs = self._build_kwargs(max_new_tokens, temperature, top_p, top_k, repeat_penalty, stream)

        try:
            if stream:
                for token in model(prompt, **kwargs):
                    yield token
            else:
                return model(prompt, **kwargs)
        except TypeError:
            # Fallback if some kwargs unsupported by the wheel
            for k in ["mirostat", "mirostat_tau", "mirostat_eta", "seed", "stop"]:
                kwargs.pop(k, None)
            if stream:
                for token in model(prompt, **kwargs):
                    yield token
            else:
                return model(prompt, **kwargs)

@st.cache_resource(show_spinner=False)
def get_hf_t2t_pipeline(model_id: str):
    return pipeline("text2text-generation", model=model_id)

def hf_t2t_generate(pipe, prompt: str, *, max_new_tokens=512, temperature=0.3, top_p=0.9):
    try:
        out = pipe(prompt, max_length=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p)
        txt = out[0]["generated_text"]
        if "Answer:" in txt:
            txt = txt.split("Answer:")[-1].strip()
        return txt
    except Exception as e:
        return f"Error generating response: {e}"