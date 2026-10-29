"""Thin client for the local OpenAI-compatible vLLM server (Qwen3-8B on :8011) used as paraphraser and judge."""
import requests, time, math
URL = "http://127.0.0.1:8011/v1/chat/completions"
MODEL = "Qwen3-8B"

def chat(messages, max_tokens=300, temperature=0.0, logprobs=False, top_logprobs=0, seed=0, retries=5):
    body = dict(model=MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature, seed=seed,
                chat_template_kwargs={"enable_thinking": False})
    if logprobs:
        body.update(logprobs=True, top_logprobs=top_logprobs)
    for a in range(retries):
        try:
            r = requests.post(URL, json=body, timeout=300)
            r.raise_for_status()
            return r.json()["choices"][0]
        except Exception as e:
            if a == retries - 1:
                raise
            time.sleep(2 * (a + 1))

def first_token_dist(choice):
    """{token_str: prob} of the first generated token from top_logprobs."""
    lp = choice["logprobs"]["content"][0]["top_logprobs"]
    return {d["token"].strip(): math.exp(d["logprob"]) for d in lp}

def label_dist(messages, labels, retries=6, top_logprobs=20):
    """Validated first-token distribution over `labels` (e.g. ["A","B","C"]).
    The shared server intermittently returned corrupted logprobs (sampled token != argmax, top-k mass ~1e-6), so a reply
    is accepted only if the sampled first token equals the top-logprob argmax and the top-k mass >= 0.5; otherwise we retry
    and finally fall back to a one-hot on the greedy text. Returns (dist, status) with status in {"lp", "text", "fail"}."""
    last = None
    for a in range(retries):
        c = chat(messages, max_tokens=3, logprobs=True, top_logprobs=top_logprobs)
        last = c
        lp = c["logprobs"]["content"][0]
        tops = {}
        for d in lp["top_logprobs"]:
            k = d["token"].strip()
            tops[k] = tops.get(k, 0.0) + math.exp(d["logprob"])
        mass = sum(tops.values())
        best = max(lp["top_logprobs"], key=lambda d: d["logprob"])["token"]
        if best == lp["token"] and mass >= 0.5:
            m = {l: tops.get(l, 0.0) for l in labels}
            s = sum(m.values())
            if s > 0:
                return {l: v / s for l, v in m.items()}, "lp"
        time.sleep(0.5 * (a + 1))
    txt = (last["message"]["content"] or "").strip()
    for l in labels:
        if txt.startswith(l):
            return {k: float(k == l) for k in labels}, "text"
    return None, "fail"
