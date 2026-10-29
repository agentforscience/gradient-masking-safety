# Datasets (downloaded 2026-10-27; raw files in datasets/raw/, not committed if large)

| file | source | use |
|---|---|---|
| raw/jbb_harmful.csv | https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/harmful-behaviors.csv | 100 harmful behaviours (E1, E4) |
| raw/jbb_benign.csv | https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/benign-behaviors.csv | 100 topic-matched benign behaviours (E3) |
| raw/harmbench.csv | https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv | 150 of 200 "standard" behaviours (seed 42, deduplicated against JBB) |
| raw/xstest.csv | https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv | 250 safe prompts with trigger words + 200 unsafe contrasts (E3) |
| raw/multijail.csv | https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail/resolve/main/multijail.csv | 315 prompts × en + 9 languages (E2) |
| raw/advbench.csv | https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv | downloaded, not used |

Derived: harmful_base.jsonl (250), controls.jsonl (550), multijail.jsonl, paraphrases(_raw).jsonl (rewrites + QC),
prompts.jsonl (5,450 target prompts: E1 750, E2 3,150, E3 550, E4 1,000).
