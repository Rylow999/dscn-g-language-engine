#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRUEBA DE CONTROL (auditoria v0.21 v8). Reentrena v0.21 v8 REAL exacto y testa
MONOSEMICAS vs POLISEMICAS del Quijote. Si las monosemicas tambien quedan
'separadas' (<85% en bucket dominante), el 39/40 es artefacto de la repulsion
incondicional, no senal de sentido. (Diagnostico de Luciano, 2026-07-28.)"""
import json, math, random, re, time
from collections import Counter
D=16; V=150; W=4; EPOCHS=8; K=2; BETA=0.10; BETA_REP=0.20; ALPHA=0.10
SEED=0
def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def norm(v): return math.sqrt(sum(x*x for x in v))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def load_seq():
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text); vocab=[w for w,_ in Counter(words).most_common(V)]
    rng=random.Random(SEED)
    idxall=[i for i,w in enumerate(words) if w in set(vocab)]
    step=max(1,len(idxall)//20000); chosen=idxall[::step][:20000]
    return [words[i] for i in chosen], vocab
def train(seq, vocab):
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    frac=[[[rng.gauss(0,1) for _ in range(D)] for _ in range(K)] for _ in range(Vn)]
    omega0=[[list(o) for o in frac[wi]] for wi in range(Vn)]
    for ep in range(EPOCHS):
        for i in range(1,len(seq)):
            a=idx[seq[i-1]]; b=idx[seq[i]]; tb=frac[b][0]
            for k in range(K):
                new=[(1-BETA)*frac[a][k][d]+BETA*tb[d] for d in range(D)]
                new=[ALPHA*omega0[a][k][d]+(1-ALPHA)*new[d] for d in range(D)]
                j=1-k; nj=norm(frac[a][j])
                if nj>1e-9: new=[new[d]-BETA_REP*(frac[a][j][d]/nj) for d in range(D)]
                frac[a][k]=new
    return frac, idx
def test_word(w, frac, idx, seq):
    if w not in idx: return None
    wi=idx[w]; occ=[i for i,x in enumerate(seq) if x==w]
    if len(occ)<20: return None
    grupos={}
    for i in occ:
        cw=list(range(max(0,i-W),i))
        if not cw: continue
        ctx=[0.0]*D
        for c in cw:
            o=frac[idx[seq[c]]][0]
            for d in range(D): ctx[d]+=o[d]
        ctx=[x/len(cw) for x in ctx]
        bestk,bestc=-1,-1e9
        for k in range(K):
            c=cos(frac[wi][k],ctx)
            if c>bestc: bestc=c; bestk=k
        grupos.setdefault(bestk,0); grupos[bestk]+=1
    n=len(occ)
    dom=max(grupos.values()) if grupos else n
    separada = (len(grupos)>=2 and dom < n*0.85)
    return dict(word=w, n=n, dom_pct=round(dom/n,3), buckets=dict(grupos), separada=separada)
def main():
    print("=== PRUEBA CONTROL v0.21 v8 (monosemicas vs polisemicas) ===")
    seq,vocab=load_seq()
    frac,idx=train(seq,vocab)
    MONO=["quijote","sancho","caballero","dia","mano","dijo","senor","casa"]
    POLI=["banco","rio","caballo","armas","cabeza","muerte","libro","gente"]
    print("\n-- MONOSEMICAS (control: deberian quedar en 1 bucket si el fix es genuino) --")
    mono_res=[]
    for w in MONO:
        r=test_word(w,frac,idx,seq)
        if r: mono_res.append(r); print(f"  {w:10} n={r['n']:4} dom={r['dom_pct']:.2f} buckets={r['buckets']} separada={r['separada']}")
    print("\n-- POLISEMICAS (deberian separarse si el fix es genuino) --")
    poli_res=[]
    for w in POLI:
        r=test_word(w,frac,idx,seq)
        if r: poli_res.append(r); print(f"  {w:10} n={r['n']:4} dom={r['dom_pct']:.2f} buckets={r['buckets']} separada={r['separada']}")
    n_mono_sep=sum(1 for r in mono_res if r['separada'])
    n_poli_sep=sum(1 for r in poli_res if r['separada'])
    print(f"\nRESULTADO: monosemicas separadas={n_mono_sep}/{len(mono_res)}  polisemicas separadas={n_poli_sep}/{len(poli_res)}")
    veredicto = ("ARTEFACTO: monosemicas tambien se separan => repulsion incondicional, no senal de sentido" 
                 if n_mono_sep>=len(mono_res)*0.5 else 
                 "GENUNO: monosemicas quedan en 1 bucket, solo polisemicas se separan")
    print("VEREDICTO:", veredicto)
    out=dict(experiment="v0.21_v8_control_audit", mono=mono_res, poli=poli_res,
             n_mono_sep=n_mono_sep, n_poli_sep=n_poli_sep, veredicto=veredicto)
    json.dump(out,open("results_v21_v8_control.json","w"),indent=2)
    print("-> results_v21_v8_control.json")
if __name__=="__main__": main()
