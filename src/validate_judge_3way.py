"""Validate the dropped 3-way A/B/C judge prompt against the hand labels."""
import sys; sys.path.insert(0, ".")
from common import *
from llm_client import label_dist
from gold_labels import LABELS
import judge, concurrent.futures as cf
P=judge.PROMPTS
G={(g['model'],g['pid']):g for m in ['qwen2.5-0.5b','qwen3-0.6b'] for g in read_jsonl(RES/f'gens/{m}.jsonl')}
gold=read_jsonl(RES/'gold_sample.jsonl')
items=[(P[x['pid']],G[(x['model'],x['pid'])]['response'],l) for x,l in zip(gold,LABELS) if l!='?']
for pid,lab in [('E1|orig|hb_chlorine_gas_safe_home_manufacture','C'),('E2|ko|mj17','C')]:
    items.append((P[pid],G[('qwen2.5-0.5b',pid)]['response'],lab))
def f(it):
    p,r,l=it
    return label_dist([{"role":"user","content":judge.J.format(req=p["request_en"],extra=judge.extra_for(p),resp=r[:1800])}],["A","B","C"])
with cf.ThreadPoolExecutor(32) as ex: out=list(ex.map(f,items))
m={'R':'A','C':'B','O':'C'}; ok=0; ok2=0
for (p,r,l),(d,st) in zip(items,out):
    pred=max(d,key=d.get); ok+=pred==m[l]; ok2+=(pred=='A')==(l=='R')
    if pred!=m[l]: print('MISS gold',l,'pred',pred,st,{k:round(v,2) for k,v in d.items()},p['pid'],r[:120].replace('\n',' '))
print('3-way acc',ok,'refusal-binary acc',ok2,'of',len(items), 'statuses', [s for _,s in out].count('lp'))
