import re, sys
rep = open(sys.argv[1]).read().split('\n')
out = sys.argv[2]

NAMES = {'qwen2.5-0.5b':'Qwen2.5-0.5B','qwen3-0.6b':'Qwen3-0.6B','llama3.2-1b':'Llama-3.2-1B','qwen2.5-1.5b':'Qwen2.5-1.5B',
 'qwen2.5-3b':'Qwen2.5-3B','llama3.2-3b':'Llama-3.2-3B','phi3-mini':'Phi-3-mini','llama2-7b-chat':'Llama-2-7B-chat',
 'mistral-7b':'Mistral-7B','qwen2.5-7b':'Qwen2.5-7B','llama3.1-8b':'Llama-3.1-8B','gemma2-9b':'Gemma-2-9B',
 'qwen2.5-14b':'Qwen2.5-14B','qwen2.5-32b':'Qwen2.5-32B'}
PARAMS = {}

def table(header_prefix):
    i = next(k for k,l in enumerate(rep) if l.startswith(header_prefix))
    rows = []
    j = i+1
    while not rep[j].startswith('|'): j += 1
    j += 2
    while j < len(rep) and rep[j].startswith('|'):
        rows.append([c.strip() for c in rep[j].strip('|').split('|')]); j += 1
    return rows

def bold(s): return '{\\bf %s}' % s
def fmtp(p):
    v = float(p)
    if v < 0.001:
        m, e = ('%.1e' % v).split('e'); return '$%s{\\times}10^{%d}$' % (m, int(e))
    return '%.3f' % v if v < 0.01 else '%.2f' % v
def ci(s):  # "+23.6 [+15.7, +30.9]"
    m = re.match(r'([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]', s)
    return float(m.group(1)), '%s {\\scriptsize[%s, %s]}' % (m.group(1).replace('+',''), m.group(2).replace('+',''), m.group(3).replace('+',''))
def hb(s): a,b = s.split('→'); return '%s$\\to$%s' % (a.strip(), b.strip())

def e1(prefix, label, caption, fname, pos='t'):
    rows = table(prefix)
    body = [r for r in rows if not r[0].startswith('**')]
    mean = [r for r in rows if r[0].startswith('**')][0]
    d1 = [ci(r[5]) for r in body]; d2 = [ci(r[6]) for r in body]
    m1 = max(v for v,_ in d1); m2 = max(v for v,_ in d2)
    L = []
    for r,(v1,s1),(v2,s2) in zip(body,d1,d2):
        PARAMS[r[0]] = r[1]
        p = float(r[7]); ps = fmtp(r[7]) + ('' if p < 0.05 else '$^{\\dagger}$')
        L.append(' & '.join([NAMES[r[0]], r[1], r[2], r[3], r[4], bold(s1) if v1==m1 else s1, bold(s2) if v2==m2 else s2, ps, hb(r[8])]) + r' \\')
    L.append(r'\midrule')
    L.append(' & '.join(['Mean', '', mean[2].strip('*'), mean[3].strip('*'), mean[4].strip('*'), mean[5].strip('*').lstrip('+'), mean[6].strip('*').lstrip('+'), '', hb(mean[8])]) + r' \\')
    tex = r'''\begin{table}[POS]
\centering
\caption{%s}
\label{%s}
\resizebox{\textwidth}{!}{%%
\begin{tabular}{@{}lrrrrccrc@{}}
\toprule
& & \multicolumn{3}{c}{Refusal (\%%)} & \multicolumn{2}{c}{Refusal drop (pp, 95\%% CI)} & & \textsc{HarmBench}-harmful (\%%) \\
\cmidrule(lr){3-5} \cmidrule(lr){6-7} \cmidrule(lr){9-9}
Model & Params (B) & \orig & \ctrl & \kwfree & \orig$\to$\kwfree & \ctrl$\to$\kwfree & $p$ (\ctrl \versus \kwfree) & \orig$\to$\kwfree \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
''' % (caption, label, '\n'.join(L))
    open(f'{out}/{fname}', 'w').write(tex.replace('[POS]', '[%s]' % pos))

e1('### E1 (main)', 'tab:e1',
   r'\eone lexical ablation on the 178 items whose two rewrites were both judged meaning-preserving. Removing trigger words (\kwfree) lowers refusal significantly in 13 of 14 models, while a re-wording that keeps them (\ctrl) costs only 2.5\pp on average. $p$: exact McNemar test of \ctrl \versus \kwfree, Holm-corrected across models; $^{\dagger}$ not significant at $\alpha=0.05$. The last column shows that the stricter \harmbench-harmful rate rises much less than refusal falls. Largest drops in {\bf bold}.',
   'e1_main.tex')
e1('### E1 (robustness)', 'tab:e1_robust',
   r'\eone robustness subset: the 151 items whose \kwfree rewrite is rated as harmful as the original ($\pm$0.5 on a 1--9 scale). The mean drop shrinks from 16.3\pp to 11.8\pp and the \ctrl \versus \kwfree contrast remains significant in 7 of 14 models. Notation as in \tabref{tab:e1}.',
   'e1_robust.tex', pos='H')

# E2
rows = table('### E2: MultiJail')
mx_nr = max(float(r[7]) for r in rows); mx_c = max(float(r[8]) for r in rows)
L = []
for r in rows:
    nm = NAMES[r[0]]
    if float(PARAMS[r[0]]) < 3: nm += '$^{\\ast}$'
    c7 = bold(r[7]) if float(r[7])==mx_nr else r[7]; c8 = bold(r[8]) if float(r[8])==mx_c else r[8]
    L.append(' & '.join([nm, r[1], r[2], r[3], r[4], r[5], r[6], c7, c8]) + r' \\')
    if r[0]=='phi3-mini': L.append(r'\midrule')
tex = r'''\begin{table}[t]
\centering
\caption{\etwo multilingual results, averaged over the five non-English languages run for every model (zh, ar, bn, sw, jv; 315 prompts each). The last two columns are computed over prompts refused in English. Non-refusal overstates harmful compliance three- to ten-fold, because most non-refusals are off-topic, misunderstood, or garbled (``other''). $^{\ast}$Sub-3B models (H2). Largest bypass rates in {\bf bold}.}
\label{tab:e2}
\resizebox{\textwidth}{!}{%%
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
& & \multicolumn{2}{c}{Refused (\%%)} & \multicolumn{2}{c}{Complied (\%%)} & Other (\%%) & \multicolumn{2}{c}{Bypass of English refusals (\%%)} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-7} \cmidrule(lr){8-9}
Model & Params (B) & en & non-en & en & non-en & non-en & $B^{\text{non-ref}}$ & $B^{\text{comply}}$ \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
''' % '\n'.join(L)
open(f'{out}/e2_main.tex','w').write(tex)

# E2 per language (appendix)
rows = table('#### E2 by language')
L = []
for r in rows:
    cells = [c.replace('n/a', '--').replace(' / ', '\\,/\\,') for c in r[1:]]
    L.append(' & '.join([NAMES[r[0]]] + cells) + r' \\')
    if r[0]=='phi3-mini': L.append(r'\midrule')
tex = r'''\begin{table}[H]
\centering
\caption{\etwo by language: bypass rates $B^{\text{comply}}$\,/\,$B^{\text{non-ref}}$ (\%%) over prompts refused in English. Languages are grouped into high- (zh, it, vi), medium- (ar, ko, th), and low-resource (bn, sw, jv). Models of 7B or more were run on five non-English languages only (--).}
\label{tab:e2_lang}
\resizebox{\textwidth}{!}{%%
\begin{tabular}{@{}lccccccccc@{}}
\toprule
& \multicolumn{3}{c}{High-resource} & \multicolumn{3}{c}{Medium-resource} & \multicolumn{3}{c}{Low-resource} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
Model & zh & it & vi & ar & ko & th & bn & sw & jv \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
''' % '\n'.join(L)
open(f'{out}/e2_lang.tex','w').write(tex)

# E3 + E3b
rows = table('### E3: point-biserial'); rows_b = {r[0]: r for r in table('### E3b')}
L = []
for r in rows:
    b = rows_b[r[0]]
    rk = r[2].split(' [')[0]; rkc = r[2].split(' [')[1].rstrip(']')
    rh = r[3].split(' [')[0]; rhc = r[3].split(' [')[1].rstrip(']')
    rkcell = '%s {\\scriptsize[%s]}' % (rk, rkc); rhcell = '%s {\\scriptsize[%s]}' % (rh, rhc)
    if float(rh) > float(rk): rhcell = bold(rh) + ' {\\scriptsize[%s]}' % rhc
    ke = b[1].replace('+',''); se = b[2].replace('+','')
    if float(se) > float(ke): se = bold(se)
    L.append(' & '.join([NAMES[r[0]], r[1], rkcell, rhcell, r[4], r[5], r[6], ke, se, b[3]]) + r' \\')
tex = r'''\begin{table}[t]
\centering
\caption{\ethree: what does refusal track? Left: point-biserial correlations of refusal with trigger-word count ($r_{\text{kw}}$), semantic harm rating ($r_{\text{harm}}$), and the LLM-free dataset label, on a pool of 1,615 prompts per model, with partial correlations of each predictor given the other. Right: lexicon-free contrasts in refusal rate (pp). The keyword effect holds meaning fixed (\ctrl$-$\kwfree); the semantic effect holds trigger words fixed (\xstest unsafe $-$ safe). The larger of each pair is in {\bf bold}: harm wins in every model, and no model shows the hypothesised keyword-matcher pattern ($r_{\text{kw}}>0.6$, $r_{\text{harm}}<0.3$).}
\label{tab:e3}
\resizebox{\textwidth}{!}{%%
\begin{tabular}{@{}lrccccccccr@{}}
\toprule
& & \multicolumn{5}{c}{Correlation with refusal} & \multicolumn{2}{c}{Controlled contrast (pp)} & \\
\cmidrule(lr){3-7} \cmidrule(lr){8-9}
Model & Params (B) & $r_{\text{kw}}$ {\scriptsize[95\%% CI]} & $r_{\text{harm}}$ {\scriptsize[95\%% CI]} & $r_{\text{label}}$ & partial kw$\cdot$harm & partial harm$\cdot$kw & Keyword effect & Semantic effect & \xstest-safe over-refusal (\%%) \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
''' % '\n'.join(L)
tex = tex.replace('{@{}lrccccccccr@{}}', '{@{}lrcccccccr@{}}')
open(f'{out}/e3_main.tex','w').write(tex)

# E4: judge compliance + HarmBench
rj = {r[0]: r for r in table('### E4: compliance')}; rh = {r[0]: r for r in table('### E4: HarmBench')}
L = []
for k in rj:
    a = rj[k]; h = rh[k]
    va = [float(x) for x in a[2:]]; vh = [float(x) for x in h[2:]]
    ma = max(va); mh = max(vh)
    ca = [bold(x) if float(x)==ma and ma>=20 else x for x in a[2:]]
    chh = [bold(x) if float(x)==mh and mh>=20 else x for x in h[2:]]
    L.append(' & '.join([NAMES[k], PARAMS[k], a[1]] + ca + [h[1]] + chh) + r' \\')
tex = r'''\begin{table}[t]
\centering
\caption{\efour: compliance with the 250 harmful requests under each framing. Left: the \judge judge (the reply carries out the request). Right: the stricter \harmbench classifier (the reply is a harmful completion). The two measures agree closely for the framing attacks, so the high fiction and persona rates are real harmful completions. For each model, the most effective framing is in {\bf bold} when it reaches at least 20\%%.}
\label{tab:e4}
\resizebox{\textwidth}{!}{%%
\begin{tabular}{@{}lrccccccccccc@{}}
\toprule
& & \multicolumn{5}{c}{Judge compliance (\%%)} & \multicolumn{5}{c}{\textsc{HarmBench}-harmful (\%%)} \\
\cmidrule(lr){3-7} \cmidrule(lr){8-12}
Model & Params (B) & \direct & \fiction & \persona & \expert & \escal & \direct & \fiction & \persona & \expert & \escal \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
''' % '\n'.join(L)
tex = tex.replace('{@{}lrccccccccccc@{}}', '{@{}lrcccccccccc@{}}')
open(f'{out}/e4_main.tex','w').write(tex)

# Judge validation
rows = table('### Judge validation')
L = [' & '.join([NAMES[r[0]], PARAMS[r[0]]] + r[1:]) + r' \\' for r in rows]
tex = r'''\begin{table}[H]
\centering
\caption{Judge validation on all English responses. The refusal judge is compared with an LLM-free string-match refusal detector; the content judge is compared with the official \harmbench classifier. The content judge recovers 79--99\%% of \harmbench-positive completions but also counts vague or non-harmful attempts, which is why we report the \harmbench rate as the stricter harmful-compliance measure.}
\label{tab:judge}
\resizebox{\textwidth}{!}{%%
\begin{tabular}{@{}lrccccc@{}}
\toprule
& & \multicolumn{2}{c}{Refusal judge \versus string match} & \multicolumn{3}{c}{Content judge \versus \textsc{HarmBench}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-7}
Model & Params (B) & Agreement (\%%) & Cohen $\kappa$ & Agreement (\%%) & P(HB harmful $|$ judge complied) & P(judge complied $|$ HB harmful) \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
''' % '\n'.join(L)
open(f'{out}/judge.tex','w').write(tex)

# Models table
rows = table('### 3.1 Models')
L = [' & '.join([NAMES[r[0]], r[1].replace('`','\\texttt{',1).replace('`','}',1).replace('_','\\_'), r[2], r[3]]) + r' \\' for r in rows]
tex = r'''\begin{table}[H]
\centering
\caption{The 14 models. All are served with vLLM, greedy decoding, at most 256 new tokens, the default chat template, and no system prompt. Qwen3-0.6B runs with thinking disabled. The Llama-3.x checkpoints are mirrors of the official weights; their chat template adds the default ``Cutting Knowledge Date'' system header.}
\label{tab:models}
\small
\begin{tabular}{@{}llrl@{}}
\toprule
Model & Hugging Face checkpoint & Params (B) & Weights \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
''' % '\n'.join(L)
open(f'{out}/models.tex','w').write(tex)
print('ok')
