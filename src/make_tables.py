"""Render the CSV result tables as markdown (results/tables.md), pasted into REPORT.md."""
import pandas as pd, numpy as np
from common import *
from models import MODELS

T = RES / "tables"
pct = lambda v: "–" if pd.isna(v) else f"{100 * v:.1f}"
ci = lambda m, lo, hi: f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
out = []

E1 = pd.read_csv(T / "E1_lexical_ablation.csv")
for sub, title in [("equivalent", "E1 (main): refusal % on items whose BOTH rewrites were judged meaning-preserving"),
                   ("equiv+harm-preserved", "E1 (robustness): items where the keyword-free rewrite is also rated as harmful as the original (±0.5)")]:
    x = E1[E1.subset == sub].sort_values("params")
    out.append(f"\n### {title} (n = {x.n.iloc[0]} items per model)\n")
    out.append("| model | params (B) | refuse orig | refuse ctrl | refuse kw-free | Δ orig→kw-free (pp, 95% CI) | Δ ctrl→kw-free (pp, 95% CI) | p ctrl vs kw-free (McNemar, Holm) | HarmBench-harmful % orig → kw-free |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in x.iterrows():
        out.append(f"| {r.model} | {r.params} | {pct(r.refuse_orig)} | {pct(r.refuse_ctrl)} | {pct(r.refuse_kwfree)} | "
                   f"{ci(r.drop_orig_kwfree, r.drop_orig_kwfree_lo, r.drop_orig_kwfree_hi)} | {ci(r.drop_ctrl_kwfree, r.drop_ctrl_kwfree_lo, r.drop_ctrl_kwfree_hi)} | "
                   f"{r.drop_ctrl_kwfree_p_holm:.3g} | {pct(r.hb_orig)} → {pct(r.hb_kwfree)} |")
    out.append(f"| **mean** | | {pct(x.refuse_orig.mean())} | {pct(x.refuse_ctrl.mean())} | {pct(x.refuse_kwfree.mean())} | {100 * x.drop_orig_kwfree.mean():+.1f} | "
               f"{100 * x.drop_ctrl_kwfree.mean():+.1f} | | {pct(x.hb_orig.mean())} → {pct(x.hb_kwfree.mean())} |")

E2 = pd.read_csv(T / "E2_multilingual.csv")
g5 = pd.read_csv(T / "E2_per_model_common5.csv").set_index("model")
order = sorted(g5.index, key=lambda m: MODELS[m]["params"])
out.append("\n### E2: MultiJail, mean over the 5 non-English languages run for every model (zh, ar, bn, sw, jv); n = 315 prompts per language\n")
out.append("| model | params (B) | refuse en | refuse non-en | comply en | comply non-en | other (off-topic/garbled) non-en | bypass: refused in en → NOT refused | bypass: refused in en → complied |")
out.append("|---|---|---|---|---|---|---|---|---|")
for m in order:
    r = g5.loc[m]
    out.append(f"| {m} | {MODELS[m]['params']} | {pct(r.refuse_en)} | {pct(r.refuse)} | {pct(r.comply_en)} | {pct(r.comply)} | {pct(r.other)} | {pct(r.bypass_nonrefusal)} | {pct(r.bypass_comply)} |")
langs = ["zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"]
out.append("\n#### E2 by language: bypass (refused in en → complied) / (refused in en → not refused), %\n")
out.append("| model | " + " | ".join(langs) + " |")
out.append("|---|" + "---|" * len(langs))
for m in order:
    x = E2[E2.model == m].set_index("lang")
    cells = [f"{pct(x.loc[l, 'bypass_comply'])} / {pct(x.loc[l, 'bypass_nonrefusal'])}" if l in x.index else "n/a" for l in langs]
    out.append(f"| {m} | " + " | ".join(cells) + " |")

E3 = pd.read_csv(T / "E3_keyword_vs_harm.csv").sort_values("params")
out.append(f"\n### E3: point-biserial correlation of refusal with keyword count vs. semantic harm (pool of {int(E3.n.iloc[0])} prompts per model)\n")
out.append("| model | params (B) | r(refusal, trigger count) [95% CI] | r(refusal, harm rating) [95% CI] | r(refusal, dataset label) | partial r kw·harm | partial r harm·kw | β kw (std.) | β harm (std.) | string-match outcome: r kw / r harm |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
for _, r in E3.iterrows():
    out.append(f"| {r.model} | {r.params} | {r.r_kw:.2f} [{r.r_kw_lo:.2f}, {r.r_kw_hi:.2f}] | {r.r_harm:.2f} [{r.r_harm_lo:.2f}, {r.r_harm_hi:.2f}] | {r.r_label:.2f} | "
               f"{r['pr_kw|harm']:.2f} | {r['pr_harm|kw']:.2f} | {r.beta_kw:.2f} | {r.beta_harm:.2f} | {r.r_kw_stringmatch:.2f} / {r.r_harm_stringmatch:.2f} |")

CT = pd.read_csv(T / "contrast_keyword_vs_semantic.csv").sort_values("params")
out.append("\n### E3b: controlled contrasts (refusal-rate differences, pp)\n")
out.append("| model | keyword effect: ctrl − kw-free (meaning fixed) | semantic effect: XSTest unsafe − safe (trigger words fixed) | XSTest-safe over-refusal % |")
out.append("|---|---|---|---|")
for _, r in CT.iterrows():
    out.append(f"| {r.model} | {100 * r.keyword_effect:+.1f} | {100 * r.semantic_effect:+.1f} | {pct(r.refuse_xstest_safe)} |")

E4 = pd.read_csv(T / "E4_roleplay_escalation.csv")
conds = ["orig", "rp_fiction", "rp_persona", "rp_expert", "escalation"]
for val, title in [("comply", "E4: compliance % (judge: reply carries out the request), n = 250 harmful requests"),
                   ("hb_harmful", "E4: HarmBench-classifier harmful-completion rate % (stricter)")]:
    piv = E4.pivot_table(index="model", columns="cond", values=val).reindex(columns=conds)
    piv = piv.loc[sorted(piv.index, key=lambda m: MODELS[m]["params"])]
    out.append(f"\n### {title}\n")
    out.append("| model | direct | fiction | persona (REX) | expert/hypothetical | 3-turn escalation |")
    out.append("|---|---|---|---|---|---|")
    for m, r in piv.iterrows():
        out.append(f"| {m} | " + " | ".join(pct(r[c]) for c in conds) + " |")

AG = pd.read_csv(T / "judge_agreement.csv")
AG = AG.loc[AG.model.map(lambda m: MODELS[m]["params"]).sort_values().index]
out.append("\n### Judge validation on English responses\n")
out.append("| model | refusal judge vs string-match agreement | Cohen κ | content judge vs HarmBench agreement | P(HB harmful | judge complied) | P(judge complied | HB harmful) |")
out.append("|---|---|---|---|---|---|")
for _, r in AG.iterrows():
    out.append(f"| {r.model} | {pct(r.refused_vs_kwref_agree)} | {r.kappa_refused_kwref:.2f} | {pct(r.complied_vs_hb_agree)} | {pct(r.p_hb_given_complied)} | {pct(r.p_complied_given_hb)} |")
open(RES / "tables.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
