"""Build the base harmful-request set (JBB-100 + 150 HarmBench standard) and the benign/XSTest control sets."""
import csv, random
from common import *

random.seed(SEED)
jbb = list(csv.DictReader(open(DATA/"raw/jbb_harmful.csv")))
hb = [r for r in csv.DictReader(open(DATA/"raw/harmbench.csv")) if r["FunctionalCategory"] == "standard"]
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
jbb_goals = {norm(r["Goal"]) for r in jbb}
hb = [r for r in hb if norm(r["Behavior"]) not in jbb_goals]
random.shuffle(hb)
harm = [dict(id=f"jbb{r['Index']}", source="JBB", category=r["Category"], topic=r["Behavior"], text=r["Goal"]) for r in jbb]
harm += [dict(id=f"hb_{r['BehaviorID']}", source="HarmBench", category=r["SemanticCategory"], topic=r["BehaviorID"].replace("_", " "), text=r["Behavior"]) for r in hb[:150]]
write_jsonl(DATA/"harmful_base.jsonl", harm)

controls = []
for r in csv.DictReader(open(DATA/"raw/xstest.csv")):
    controls.append(dict(id=f"xs{r['id']}", source="XSTest", category=r["type"], label=r["label"], text=r["prompt"]))
for r in csv.DictReader(open(DATA/"raw/jbb_benign.csv")):
    controls.append(dict(id=f"jbbb{r['Index']}", source="JBB-benign", category=r["Category"], label="safe", text=r["Goal"]))
write_jsonl(DATA/"controls.jsonl", controls)

mj = list(csv.DictReader(open(DATA/"raw/multijail.csv", encoding="utf-8-sig")))
write_jsonl(DATA/"multijail.jsonl", mj)
print(len(harm), "harmful;", len(controls), "controls;", len(mj), "multijail; langs:", [k for k in mj[0] if len(k) == 2])
print("mean trigger count harmful:", sum(trigger_count(h["text"]) for h in harm)/len(harm))
print("frac harmful w/ >=1 trigger:", sum(trigger_count(h["text"])>0 for h in harm)/len(harm))
