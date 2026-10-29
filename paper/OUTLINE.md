# Outline: Fiction, Not Vocabulary

Title: Fiction, Not Vocabulary: Safety Tuning Leaves Framing Blind Spots, Not Keyword Blind Spots, in Open-Weight Chat Models

## Abstract
Problem (gradient-masking analogy) -> gap (no controlled separation of keyword vs paraphrase vs meaning across a size ladder)
-> approach (14 models, 0.5-32.8B, 4 shifts on same 250 items, matched ctrl rewrite, two judges)
-> results (16.3pp lexical leak, 2.5pp for ctrl; r_harm 0.27-0.63 vs r_kw 0.10-0.25; fiction 51-78%, persona 86-93%;
multilingual non-refusal 44-78% but compliance 6.6-14.8%) -> significance (red-team framing, measure harmful compliance).

## 1 Introduction
- Hook: safety tuning is the main defence; is it semantic or surface?
- Analogy to gradient masking (Athalye et al.) -> hypotheses H1-H4.
- Gap: effects studied separately; no keyword-kept control; no same-item size ladder.
- Approach: Fig 1 (four shifts x 14 models x judges).
- Preview: numbers above.
- Contributions (4 bullets).

## 2 Related work
Jailbreaks via reformulation; multilingual; role-play & multi-turn; exaggerated safety; mechanisms of safety failure; evaluation.

## 3 Method
3.1 Problem formulation (refusal indicator, effects Delta_rw, Delta_kw, bypass, correlations)
3.2 Models (Table in appendix)  3.3 Data per experiment  3.4 Labelling  3.5 Statistics  3.6 Implementation

## 4 Results
Table: hypothesis verdicts. E1 (Fig+Table), E2 (Fig+Table), E3 (Fig+Table), E4 (Fig+Table), judge validation.

## 5 Discussion
Interpretation (framing > vocabulary), implications for red-teaming / multilingual measurement, limitations, broader impact.

## 6 Conclusion

## Appendix
Models table, E1 robustness table, E2 per-language table, judge validation table, qualitative examples, prompt templates.
