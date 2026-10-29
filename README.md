# Do safety-tuned LLMs refuse on keywords or on meaning?

We test whether safety tuning in open chat models (0.5B–32B; Qwen2.5, Qwen3, Llama-3.x, Llama-2, Phi-3, Mistral,
Gemma-2) works like "gradient masking": a defence keyed to surface patterns that simple distribution shifts get around.
Every model gets the same 250 harmful requests (JailbreakBench + HarmBench) in four forms: rewrites with the trigger
words removed vs. kept, 10 languages (MultiJail), role-play framings, and a 3-turn escalation. Refusal is also tested
against keyword presence vs. a semantic harm rating, using XSTest prompts where the two diverge.
All outputs come from real models (vLLM, greedy decoding).

**Full write-up: [REPORT.md](REPORT.md)**. Plan: [planning.md](planning.md).

## Key findings (14 models)
See REPORT.md §1 for details.
- **Keyword removal works, but not as strongly as hypothesised.** Rewriting a harmful request without its trigger words
  lowers refusal by 4.5–27.5 pp (mean 16.3 pp; significant in 13/14 models; only 3 models fall in the hypothesised
  20–40 pp band). A plain re-wording that keeps the trigger words costs only 2.5 pp. On items where rated harm is
  preserved, the drop is 11.8 pp. HarmBench-harmful completions rise by only 3.5 pp. The effect does **not** shrink
  with model size.
- **Multilingual "bypass" is mostly incoherence, not jailbreak.** In sub-3B models, 44–78% of prompts refused in English
  are not refused in another language. But only 6.6–14.8% of them get a reply that actually carries out the request;
  the rest are off-topic or garbled, especially in Bengali, Swahili and Javanese. Larger models comply more often in
  low-resource languages.
- **Refusal tracks meaning more than keywords (H3 refuted).** In every model, r(refusal, trigger-word count) is
  0.10–0.25, while r(refusal, semantic harm rating) is 0.27–0.63. With trigger words held fixed (XSTest unsafe vs.
  safe), refusal differs by 44–82 pp; with meaning held fixed (keywords kept vs. removed), by only 6–21 pp.
- **Role-play blind spots are large and family-specific (H4 supported for role-play; not a scale effect).** Fiction
  framing raises compliance from ≤11% to 51–78% for Qwen2.5-1.5B/3B/7B and Phi-3. A "no-rules persona" raises
  Llama-3.1-8B to 93% and Gemma-2-9B to 86% (both ≤14% when asked directly). Within Qwen2.5, fiction compliance is non-monotonic
  (7B 71% → 14B 11% → 32B 40%). Llama-2-7B-chat and Llama-3.2-1B/3B resist every framing. 3-turn escalation is a weaker attack (≤24% for aligned models ≥1.5B).

## Reproduce
```bash
uv venv --python 3.12 && source .venv/bin/activate && uv sync      # vllm 0.32, pandas, scipy, statsmodels, matplotlib
cd src
python prep_data.py            # datasets/raw/*.csv -> harmful_base / controls / multijail jsonl
python paraphrase.py           # keyword-free + control rewrites (needs an OpenAI-compatible Qwen3-8B at :8011)
python build_prompts.py        # datasets/prompts.jsonl (5,450 prompts)
python rate_prompts.py         # paraphrase equivalence + semantic harm ratings (Qwen3-8B)
echo 30 > ../logs/hold_g1.target; CUDA_VISIBLE_DEVICES=0 python ../tools/holder2.py ../logs/hold_g1.target ../logs/hold_g1.actual &
CUDA_VISIBLE_DEVICES=0 python worker.py ../logs/queue.txt   # takes over the held memory; queue lines like task=gen:qwen2.5-7b, task=hbcls
./run_judges.sh                # Qwen3-8B judge (refusal + content questions); or task=judge_local in the worker
python analysis.py             # tables -> results/tables, figures -> figures/, results/summary.json
python examples.py             # qualitative examples -> results/examples.md
```
Raw datasets come from GitHub/HF (URLs in `datasets/README.md`). Models come from the local HF cache
(`HF_HOME=/data/transformers`).

## Layout
```
src/          common.py (paths, trigger lexicon) · models.py (14 targets) · prep_data/paraphrase/build_prompts/rate_prompts
              worker.py + tasks.py (in-process vLLM worker for a shared GPU; generation, HarmBench classifier, local judge)
              judge.py, run_judges.sh, llm_client.py (validated log-prob judge) · validate_judge*.py, gold_labels.py
              analysis.py, examples.py
datasets/     raw/ (downloads), prompts.jsonl, paraphrases.jsonl, ...
results/      gens/ (raw model outputs — contain harmful text), judged/, hbcls/, tables/, summary.json, examples.md
figures/      e1_refusal_by_rewrite.png, e2_multilingual_*.png, e3_keyword_vs_harm.png, e4_roleplay_escalation.png
```
**Content warning:** `results/gens/` holds unfiltered model outputs to harmful requests, kept for research and auditing.
