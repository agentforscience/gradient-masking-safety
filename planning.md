# Planning: Do Alignment Methods Create Exploitable Blind Spots? ("Gradient masking" in safety-tuned LLMs)

## Motivation & Novelty Assessment

### Why This Research Matters
Safety tuning (SFT on refusals, RLHF/DPO) is the main defence deployed open-weight chat models have against misuse.
If what it learns is a *lexical* refusal reflex ("bomb" → refuse) and not a *semantic* notion of harm, then safety
is brittle in the way adversarially-trained classifiers with gradient masking are: the defence looks robust on the
inputs it was tuned on, but simple distribution shifts (re-wording, translation, role-play, multi-turn build-up)
go around it, while benign prompts that contain trigger words get refused (over-refusal). This matters to everyone who
ships small on-device models, because those get the least safety data and compute.

### Gap in Existing Work
(The resource-finder stage was skipped for this workspace, so there was no pre-built literature review; this is from
background knowledge.) Prior work shows each effect separately: multilingual jailbreaks (Deng et al. 2024,
*MultiJail*; Yong et al. 2023, low-resource languages), past-tense/paraphrase reformulations (Andriushchenko &
Flammarion 2024), exaggerated safety on trigger words (Röttger et al. 2024, *XSTest*), role-play/persona jailbreaks
(Shen et al. 2024, *DAN*) and multi-turn escalation (Russinovich et al. 2024, *Crescendo*). Missing:
(i) a **controlled lexical ablation**, where a trigger-word-free rewrite is compared with a *control* rewrite that keeps
the trigger words, so a "paraphrase effect" is separated from a "keyword effect"; (ii) a direct, per-model estimate of
how much refusal tracks **keyword presence vs. a semantic harm rating** on a prompt pool where the two are
deliberately decorrelated; (iii) all of this measured on the **same harmful items across a 0.5B–32B ladder** of
current open models.

### Our Novel Contribution
A single-harness, 14-model study (0.5B–32B; Qwen2.5, Qwen3, Llama-3.x, Llama-2, Phi-3, Mistral, Gemma-2) that measures
refusal under four kinds of distribution shift on the same items, using a keyword-ablation design with a matched control,
and a regression of refusal on keyword count vs. semantic harm, per model.

### Experiment Justification
- **E1 Lexical ablation (paraphrase without trigger words)**: tests H1 (20–40 pp refusal drop). The `ctrl` rewrite
  keeps trigger words, so `orig→ctrl` is the cost of re-wording alone and `ctrl→kwfree` is the cost of losing the
  keywords.
- **E2 Multilingual (MultiJail, 10 languages)**: tests H2 (sub-3B models bypassed 40–60% of the time). Bypass is split
  into *harmful compliance* and *non-refusal*, because small models often produce off-topic or garbled text in
  low-resource languages, which is not a real bypass.
- **E3 Keyword vs. semantic harm**: tests H3 (r_keyword > 0.6, r_harm < 0.3 in small models). The pool mixes harmful
  originals, keyword-free harmful rewrites, XSTest safe prompts (trigger words, benign meaning) and XSTest/JBB contrasts,
  so keywords and harm are partly decorrelated.
- **E4 Role-play and escalation**: tests H4 (large models generalise semantically but still have role-play and
  escalation blind spots). Three role-play templates plus a 3-turn escalation in which the model's own earlier answers
  stay in context.

## Research Question
Do safety-tuned LLMs refuse on the basis of surface features (trigger words, English, direct phrasing) and not the
meaning of harm, and does this get better with scale?

## Hypothesis Decomposition
- H1: refusal(orig) − refusal(kwfree) is 20–40 pp, and larger than refusal(orig) − refusal(ctrl).
- H2: for models < 3B, the non-English bypass rate (harmful compliance on MultiJail prompts that are refused in English)
  is 40–60%.
- H3: in small models, corr(refusal, keyword count) > 0.6 and corr(refusal, semantic harm) < 0.3; in larger models the
  pattern reverses.
- H4: larger models have a smaller keyword gap (E1) and a higher harm correlation (E3), but E4 role-play/escalation
  compliance stays substantial.

## Proposed Methodology
### Approach
Real local models served with vLLM (no API key on this host; OpenRouter unavailable), greedy decoding, max 256 new
tokens, default chat template and no system prompt. Responses are labelled by Qwen3-8B (a separate model, not among the
targets) as refusal / compliance / neither, using first-token probabilities. A string-match refusal detector and the
HarmBench classifier (if GPU time allows) are used for cross-checks.

### Data
- Harmful base set: JBB-Behaviors (100) + 150 HarmBench "standard" behaviours (seed 42, deduplicated) = 250.
- Rewrites by Qwen3-8B: `kwfree` (no alarming words, no added framing) and `ctrl` (re-worded, keeps the alarming words).
  Rewrites are QC'd with Qwen3-8B paraphrase-equivalence P(same) and with the a-priori trigger lexicon in `src/common.py`.
- MultiJail: 315 prompts × {en, zh, it, vi, ar, ko, th, bn, sw, jv}.
- Controls: XSTest (250 safe + 200 unsafe contrast), JBB benign (100).
- Semantic harm rating: Qwen3-8B, 1–9, probability-weighted, instructed to judge meaning rather than wording.
  Dataset labels (harmful/safe) are used as a second semantic measure that does not depend on an LLM.

### Baselines
- Within-item baselines: `orig` (E1), `en` (E2), single-turn `orig` (E4).
- Lexical baseline: a keyword-only refusal predictor; how well the trigger count alone predicts refusal.

### Evaluation Metrics
Refusal rate, compliance rate, other rate; bypass rate = P(comply in condition | refused in baseline);
point-biserial / Spearman correlations; logistic regression of refusal on standardised keyword count and harm rating.

### Statistical Analysis Plan
Paired item-level comparisons: McNemar exact test for refusal changes; 95% bootstrap CIs (2,000 resamples over items);
Holm correction across models within each hypothesis; α = 0.05. Correlations are reported with bootstrap CIs.
Scale trends: Spearman correlation of the effect with log(params) across models.

## Expected Outcomes
Support for H1: kwfree drop of 20–40 pp with the ctrl drop much smaller. Refutation: drop < 10 pp, or drop ≈ ctrl drop.
H2: support if bypass in low-resource languages for < 3B models is 40–60% under the harmful-compliance definition.
H3: support if the keyword correlation exceeds the harm correlation in small models.

## Timeline
Setup/data 45 min → generation (14 models, ~5.4k prompts each) ~1.5 h on a held 34 GB slice of a shared A6000,
judged in parallel → analysis 1 h → report 45 min.

## Potential Challenges
- Shared GPUs (other tenants); handled with a single in-process vLLM worker that keeps its memory between models.
- LLM-judge bias on harmful content (seen already: a first equivalence prompt made Qwen3-8B answer "No" on identical
  harmful pairs); mitigated with neutral framing, a string-match cross-check and manual spot checks.
- Rewrites that drift in meaning: primary analyses use rewrites judged equivalent (P(same) ≥ 0.5); all-items results
  reported as sensitivity.
- Keyword lexicon circularity: the lexicon is fixed a priori; dataset labels are used as an LLM-free harm measure.

## Success Criteria
All four experiments run on ≥ 10 models with real generations, each hypothesis gets a quantitative verdict with CIs,
and the report states clearly which parts are supported and which are refuted.
