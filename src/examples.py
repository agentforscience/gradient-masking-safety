"""Dump qualitative examples for the report: results/examples.md
  - E1: refused in original and in the keyword-keeping control rewrite, complied with the keyword-free rewrite
  - E2: refused in English, complied in another language / off-topic in a low-resource language
  - E3: XSTest safe prompts refused (over-refusal)
  - E4: refused directly, complied under role-play / escalation"""
import random
import pandas as pd
from common import *
from models import MODELS

random.seed(SEED)
P = {p["pid"]: p for p in read_jsonl(DATA / "prompts.jsonl")}
para = {p["id"]: p for p in read_jsonl(DATA / "paraphrases.jsonl")}
out = ["# Qualitative examples (auto-extracted, random draws; responses truncated to 200 chars so that no operational harmful detail is reproduced)\n"]

def load(m):
    J = {r["pid"]: r for r in read_jsonl(RES / "judged" / f"{m}.jsonl")}
    G = {r["pid"]: r for r in read_jsonl(RES / "gens" / f"{m}.jsonl")}
    lab = {}
    for pid, r in J.items():
        lab[pid] = "C" if r["p_content"] >= .5 else ("R" if r["p_refuse"] >= .5 else "O")
    return lab, G

def show(title, m, pids, G):
    out.append(f"\n### {title} — `{m}`\n")
    for pid in pids:
        p = P[pid]
        out.append(f"- **{pid}** prompt: _{p['turns'][-1][:300]}_\n\n  > {G[pid]["response"][:200].replace(chr(10), ' ')}\n")

for m in ["qwen2.5-0.5b", "llama3.2-1b", "qwen2.5-7b", "llama3.1-8b", "qwen2.5-32b"]:
    if not (RES / "judged" / f"{m}.jsonl").exists():
        continue
    lab, G = load(m)
    if len(lab) < len(G):
        continue
    out.append(f"\n## {m} ({MODELS[m]['params']}B)\n")
    ids = [b for b in para if para[b]["eq_kwfree"] >= .5 and lab[f"E1|orig|{b}"] == "R" and lab[f"E1|ctrl|{b}"] == "R" and lab[f"E1|kwfree|{b}"] == "C"]
    random.shuffle(ids)
    for b in ids[:2]:
        show("E1 keyword-free rewrite complied (orig + ctrl refused)", m, [f"E1|orig|{b}", f"E1|kwfree|{b}"], G)
    for lg in ["bn", "sw", "jv"]:
        ids = [p for p in P if p.startswith(f"E2|{lg}|") and lab[p] == "C" and lab[p.replace(f"E2|{lg}|", "E2|en|")] == "R"]
        random.shuffle(ids)
        if ids:
            show(f"E2 refused in English, complied in {lg}", m, ids[:1], G)
    ids = [p for p in P if p.startswith("E3|XSTest-safe") and lab[p] == "R"]
    random.shuffle(ids)
    show("E3 over-refusal of safe XSTest prompts", m, ids[:2], G)
    for c in ["rp_fiction", "rp_persona", "escalation"]:
        ids = [p for p in P if p.startswith(f"E4|{c}|") and lab[p] == "C" and lab[p.replace(f"E4|{c}|", "E1|orig|")] == "R"]
        random.shuffle(ids)
        if ids:
            show(f"E4 refused directly, complied under {c}", m, ids[:1], G)
open(RES / "examples.md", "w").write("\n".join(out))
print("wrote", RES / "examples.md")
