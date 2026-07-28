#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.23 — COMPOSICION RELACIONAL (Gap 2 hacia pseudoAGI).
El grafo fractal (v0.21 v8) codifica CO-OCURRENCIA, no RELACION ESTRUCTURADA.
Esto aprende TRIPLAS (sujeto, RELACION, objeto) por Hebb 3-body sobre patrones
ordenados: cuando 'banco cuenta dinero' ocurre, refuerza el borde (banco, TIENE,
dinero) via la co-ocurrencia ordenada (no solo adyacencia).
Test honesto: tras entrenar, 'banco ___ dinero' debe completar con relacion de
TIENE/contener mejor que azar; y 'banco ___ rio' con relacion de lugar/bordear.
"""
import json, math, random, re, time
from collections import Counter, defaultdict
D=16; WIN=4; SEED=0
def norm(v): return math.sqrt(sum(x*x for x in v)) or 1e-9
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def build_corpus(seed=SEED, n=30):
    rng=random.Random(seed)
    # (sujeto, [objetos de relacion TIENE], [objetos de relacion LUGAR])
    triples={
      "banco": (["dinero","cuenta","oro"], ["rio","plaza","ciudad"]),
      "casa":  (["puerta","techo","familia"], ["calle","barrio","campo"]),
      "arbol": (["hoja","fruto","rama"], ["bosque","suelo","jardin"]),
    }
    filler=["el","la","de","y","en","con","por","un","una","que","los","las"]
    seq=[]; rel=[]  # rel marca la relacion del par (suj,obj) inmediato
    for s,(ht,lp) in triples.items():
        for _ in range(n):
            o=rng.choice(ht); seq+=["el",s,"tiene",o]; rel+=["X","TIENE","TIENE","TIENE"]
        for _ in range(n):
            o=rng.choice(lp); seq+=["el",s,"esta_en",o]; rel+=["X","LUGAR","LUGAR","LUGAR"]
    pr=list(zip(seq,rel)); rng.shuffle(pr); seq=[p[0] for p in pr]; rel=[p[1] for p in pr]
    vocab=list(dict.fromkeys(seq))
    return seq,rel,vocab,list(triples.keys()),triples
def train_rel(seq,rel,vocab,epochs=8):
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}; rng=random.Random(SEED)
    # embedding por palabra
    emb=[[rng.gauss(0,1) for _ in range(D)] for _ in range(Vn)]
    o0=[[x for x in e] for e in emb]
    # matriz de relacion aprendida: R[r] (DxD) para r en {TIENE, LUGAR}
    R={"TIENE":[[1.0 if i==j else 0.01*rng.gauss(0,1) for j in range(D)] for i in range(D)],
       "LUGAR":[[1.0 if i==j else 0.01*rng.gauss(0,1) for j in range(D)] for i in range(D)]}
    for ep in range(epochs):
        for i in range(2,len(seq)):
            if rel[i] in ("TIENE","LUGAR") and seq[i-2] in idx and seq[i] in idx:
                s=idx[seq[i-2]]; o=idx[seq[i]]; r=rel[i]
                # Hebb 3-body: refuerza que R[r]*emb[s] ~ emb[o]
                ps=emb[s]; po=emb[o]
                psr=[sum(R[r][k][d]*ps[d] for d in range(D)) for k in range(D)]
                po_n=[x/norm(po) for x in po]; psr_n=[x/norm(psr) for x in psr]
                # R[r] += lr * outer(psr_n, po_n)
                lr=0.01
                R[r]=[[R[r][a][b]+lr*psr_n[a]*po_n[b] for b in range(D)] for a in range(D)]
                # tambien acerca emb[s],emb[o] (asociacion basica)
                new_s=[0.9*emb[s][d]+0.1*po[d] for d in range(D)]
                emb[s]=[0.9*o0[s][d]+0.1*new_s[d] for d in range(D)]
                new_o=[0.9*emb[o][d]+0.1*ps[d] for d in range(D)]
                emb[o]=[0.9*o0[o][d]+0.1*new_o[d] for d in range(D)]
    return emb,R,idx
def predict_rel(emb,R,idx,s,o):
    # ¿que relacion mejor conecta s y o? score = cos(R[r]*emb[s], emb[o])
    best=None; bs=-2.0
    for r in R:
        psr=[sum(R[r][k][d]*emb[idx[s]][d] for d in range(D)) for k in range(D)]
        sc=cos(psr,emb[idx[o]])
        if sc>bs: bs=sc; best=r
    return best,round(bs,3)
def main():
    print("=== v0.23 COMPOSICION RELACIONAL (Hebb 3-body) ===")
    t0=time.time()
    seq,rel,vocab,subs,triples=build_corpus()
    emb,R,idx=train_rel(seq,rel,vocab)
    # TEST: para cada sujeto, predecir relacion hacia un objeto TIENE y uno LUGAR
    test={}
    for s,(ht,lp) in triples.items():
        for o in ht[:2]:
            r,sc=predict_rel(emb,R,idx,s,o); test[f"{s}-{o}"]=dict(pred=r,score=sc,gt="TIENE")
        for o in lp[:2]:
            r,sc=predict_rel(emb,R,idx,s,o); test[f"{s}-{o}"]=dict(pred=r,score=sc,gt="LUGAR")
    ok=sum(1 for v in test.values() if v["pred"]==v["gt"]); tot=len(test)
    # baseline azar: 2 relaciones -> 0.5
    print(f"train+eval {time.time()-t0:.0f}s")
    print(f"test relacional: {ok}/{tot} = {round(ok/tot,3)} (azar=0.5)")
    print("detalle:", json.dumps(test,ensure_ascii=False))
    out=dict(experiment="v0.23_composicion_relacional",
             hypothesis="Hebb 3-body aprende relaciones (TIENE/LUGAR) como matrices R[r]: tras entrenar, predecir la relacion correcta entre sujeto y objeto debe superar el azar (0.5).",
             params=dict(d=D,window=WIN,epochs=8),
             test_relacional=dict(acc=round(ok/tot,3),n=tot,baseline_azar=0.5),
             supera_azar=ok/tot>0.5,
             detalle=test)
    json.dump(out,open("results_v23.json","w"),indent=2)
    print("\n-> results_v23.json")
if __name__=="__main__": main()
