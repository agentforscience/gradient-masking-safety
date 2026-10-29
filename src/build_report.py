"""Fill REPORT_body.md placeholders from results/tables/*.csv + results/summary.json and write REPORT.md."""
import json
import pandas as pd, numpy as np
from common import *
from models import MODELS, SMALL

T = RES / "tables"
S = json.load(open(RES / "summary.json"))
E1 = pd.read_csv(T / "E1_lexical_ablation.csv")
eq = E1[E1.subset == "equivalent"]
hp = E1[E1.subset == "equiv+harm-preserved"]
g5 = pd.read_csv(T / "E2_per_model_common5.csv").set_index("model")
E3 = pd.read_csv(T / "E3_keyword_vs_harm.csv")
CT = pd.read_csv(T / "contrast_keyword_vs_semantic.csv")
models = list(eq.model)
small = [m for m in models if MODELS[m]["params"] < 3.0]
rng_ = lambda s, f=100, d=0: f"{f * s.min():.{d}f}–{f * s.max():.{d}f}"

def model_table():
    rows = ["| key | HF checkpoint | params (B) | weights |", "|---|---|---|---|"]
    for m in sorted(models, key=lambda k: MODELS[k]["params"]):
        c = MODELS[m]
        q = "AWQ-INT4" if "AWQ" in c["hf"] else "bf16"
        rows.append(f"| {m} | `{c['hf']}` | {c['params']} | {q} |")
    return "\n".join(rows)

rho = S["E1"]["spearman_drop_vs_logparams"]
E4 = pd.read_csv(T / "E4_roleplay_escalation.csv")
fic = E4[(E4.cond == "rp_fiction") & E4.model.str.startswith("qwen2.5")].sort_values("params")
qwen_trend = "fiction compliance " + " → ".join(f"{r.params:g}B {100 * r.comply:.0f}%" for _, r in fic.iterrows())
q32 = E4[E4.model == "qwen2.5-32b"]
q32_note = (f", and Qwen2.5-32B stays ≤ {100 * q32[q32.cond != 'orig'].comply.max():.0f}% under every framing" if len(q32) else "")
E2 = pd.read_csv(T / "E2_multilingual.csv")
AG = pd.read_csv(T / "judge_agreement.csv")
fill = {
    "HP_NSIG": f"{int((hp.drop_ctrl_kwfree_p_holm < 0.05).sum())}/{len(hp)} (vs {int((eq.drop_ctrl_kwfree_p_holm < 0.05).sum())}/{len(eq)})",
    "EN_REF_RANGE": rng_(E2[E2.lang == "en"].refuse),
    "AG_SM": rng_(AG.refused_vs_kwref_agree), "AG_KAPPA": rng_(AG.kappa_refused_kwref, 1, 2),
    "AG_HB": rng_(AG.complied_vs_hb_agree.dropna()), "AG_HB_RECALL": rng_(AG.p_complied_given_hb.dropna()),
    "NMODELS": str(len(models)),
    "MAXB": f"{max(MODELS[m]['params'] for m in models):g}",
    "E1_MEAN": f"{100 * eq.drop_orig_kwfree.mean():.1f}",
    "E1_RANGE": rng_(eq.drop_orig_kwfree, d=1),
    "E1_NSIG": f"{int((eq.drop_orig_kwfree_p_holm < 0.05).sum())}/{len(eq)}",
    "E1_CTRL": f"{100 * eq.drop_orig_ctrl.mean():.1f}",
    "E1_N2040": str(int(((eq.drop_orig_kwfree >= .2) & (eq.drop_orig_kwfree <= .4)).sum())),
    "E1_HP": f"{100 * hp.drop_orig_kwfree.mean():.1f}",
    "E1_HB": f"{100 * (eq.hb_kwfree - eq.hb_orig).mean():.1f}",
    "E1_HB_FROM": f"{100 * eq.hb_orig.mean():.1f}",
    "E1_HB_TO": f"{100 * eq.hb_kwfree.mean():.1f}",
    "E1_KWFREE_REF": f"{100 * eq.refuse_kwfree.mean():.0f}",
    "E1_RHO": f"{rho['statistic']:.2f} (p = {rho['pvalue']:.2f})",
    "E2_NONREF": rng_(g5.loc[small, "bypass_nonrefusal"]) + "%",
    "E2_COMPLY": rng_(g5.loc[small, "bypass_comply"], d=1) + "%",
    "E2_LRL_SMALL": f"{100 * S['E2']['by_resource_small']['bypass_comply']['LRL']:.1f}%",
    "E2_LRL_LARGE": f"{100 * S['E2']['by_resource_large']['bypass_comply']['LRL']:.1f}%",
    "E3_KW": rng_(E3.r_kw, 1, 2),
    "E3_HARM": rng_(E3.r_harm, 1, 2),
    "CT_SEM": rng_(CT.semantic_effect),
    "CT_KW": rng_(CT.keyword_effect),
    "QMAX": f"{max(MODELS[m]['params'] for m in models if m.startswith('qwen2.5')):g}",
    "LLAMA2_NOTE": ("included (run after the other models)." if "llama2-7b-chat" in models else
                    "queued last; generation/judging did not finish within the session budget, so it is not in the results."),
    "MODEL_TABLE": model_table(),
    "QWEN_FICTION_TREND": qwen_trend,
    "QWEN32_NOTE": q32_note,
    "TABLES": open(RES / "tables.md").read(),
}
txt = open(ROOT / "REPORT_body.md").read()
for k, v in fill.items():
    txt = txt.replace("{{" + k + "}}", v)
assert "{{" not in txt, [l for l in txt.splitlines() if "{{" in l]
open(ROOT / "REPORT.md", "w").write(txt)
print(json.dumps({k: v for k, v in fill.items() if k not in ("TABLES", "MODEL_TABLE")}, indent=1))
