"""Target-model registry: key -> (HF id, approx params in B, need_GB, kv_GB, extra LLM kwargs)."""
MODELS = {
    "qwen2.5-0.5b":   dict(hf="Qwen/Qwen2.5-0.5B-Instruct", params=0.5, need=5, kv=2.0),
    "qwen3-0.6b":     dict(hf="Qwen/Qwen3-0.6B", params=0.6, need=5, kv=2.0, thinking_off=True),
    "llama3.2-1b":    dict(hf="unsloth/Llama-3.2-1B-Instruct", params=1.2, need=6, kv=2.0),
    "qwen2.5-1.5b":   dict(hf="Qwen/Qwen2.5-1.5B-Instruct", params=1.5, need=6, kv=2.0),
    "qwen2.5-3b":     dict(hf="Qwen/Qwen2.5-3B-Instruct", params=3.1, need=10, kv=3.0),
    "llama3.2-3b":    dict(hf="unsloth/Llama-3.2-3B-Instruct", params=3.2, need=10, kv=2.5),
    "phi3-mini":      dict(hf="microsoft/Phi-3-mini-4k-instruct", params=3.8, need=13, kv=4.0),
    "mistral-7b":     dict(hf="solidrust/Mistral-7B-Instruct-v0.3-AWQ", params=7.2, need=9, kv=3.0),
    "llama2-7b-chat": dict(hf="meta-llama/Llama-2-7b-chat-hf", params=6.7, need=20, kv=4.5),
    "qwen2.5-7b":     dict(hf="Qwen/Qwen2.5-7B-Instruct", params=7.6, need=19, kv=3.0),
    "llama3.1-8b":    dict(hf="unsloth/Meta-Llama-3.1-8B-Instruct", params=8.0, need=20, kv=3.0),
    "gemma2-9b":      dict(hf="hugging-quants/gemma-2-9b-it-AWQ-INT4", params=9.2, need=11, kv=3.0),
    "qwen2.5-14b":    dict(hf="Qwen/Qwen2.5-14B-Instruct-AWQ", params=14.8, need=15, kv=3.0),
    "qwen2.5-32b":    dict(hf="Qwen/Qwen2.5-32B-Instruct-AWQ", params=32.8, need=26, kv=4.0),
}
SMALL = [k for k, v in MODELS.items() if v["params"] < 3.5]
