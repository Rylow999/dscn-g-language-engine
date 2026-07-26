#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.7-final — CONTEXTO LIMPIO. omega FIJOS (no se contaminan).
Solo un estado aparte omega_ctx[(w_prev,w_curr)] aprende a predecir w_next.
Es un modelo de TRIGRAMA en espacio de embeddings. Test honesto: ¿el contexto
(2 palabras) sube accuracy vs v0.6a (bigrama, W=1, 0.1011)?
omega fijos = representacion; omega_ctx = lo unico que aprende.
"""
import json, math, random, re, sys, time
from collections import defaultdict
D=8; ALPHA=5.0; BETA=0.20; V=150; EPOCHS=2

def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words, V):
    from collections import Counter
    return [w for w,_ in Counter(words).most_common(V)]
def omega_of(w, rng):
    r=random.Random(hash(w)%100000); return [r.gauss(0,1) for _ in range(D)]
def norm(v): return math.sqrt(sum(x*x for x in v))
def affinity(q,w):
    d=math.sqrt(sum((a-b)**2 for a,b in zip(q,w))); return math.exp(-ALPHA*d)

def main():
    print("=== v0.7-final CONTEXTO LIMPIO (trigrama, omega fijos) ===")
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text); vocab=build_vocab(words,V)
    rng=random.Random(0)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]  # FIJOS
    idx={w:i for i,w in enumerate(vocab)}
    seq=[w for w in words if w in set(vocab)]
    # omega_ctx por par (prev,curr) -> aprende hacia omega[next]
    omega_ctx=defaultdict(lambda: [rng.gauss(0,1) for _ in range(D)])
    t0=time.time()
    for ep in range(EPOCHS):
        for i in range(len(seq)-2):
            a,b,c=seq[i],seq[i+1],seq[i+2]
            if a not in idx or b not in idx or c not in idx: continue
            key=(a,b)
            oc=omega_ctx[key]
            on=omega[idx[c]]
            omega_ctx[key]=[(1-BETA)*oc[k]+BETA*on[k] for k in range(D)]
    print(f"entrenado {time.time()-t0:.0f}s")
    # evaluar: dado par (prev,curr), predecir next por max afinidad de omega_ctx
    ok=tot=0
    for i in range(len(seq)-2):
        a,b,c=seq[i],seq[i+1],seq[i+2]
        if a not in idx or b not in idx or c not in idx: continue
        key=(a,b); oc=omega_ctx[key]
        bestw,bests=-1,-1.0
        for j,o in enumerate(omega):
            s=affinity(oc,o)
            if s>bests: bests=s; bestw=j
        if bestw==idx[c]: ok+=1
        tot+=1
    acc=ok/tot if tot else 0
    out=dict(experiment="v0.7_final_contexto_limpio",
             hypothesis="Contexto de 2 palabras (trigrama) sube accuracy vs v0.6a (bigrama, 0.1011). omega fijos.",
             params=dict(d=D,alpha=ALPHA,beta=BETA,V=V,epochs=EPOCHS,contexto="2 palabras",corpus="don_quijote"),
             accuracy=round(acc,4),
             comparar_con_v06a=0.1011,
             nota="omega_ctx[(prev,curr)] aprende; omega nodos FIJOS (no se tocan).")
    with open("results_v07final.json","w") as f: json.dump(out,f,indent=2)
    print(f"accuracy trigrama (ctx 2 pal): {acc:.4f}  (v0.6a bigrama: 0.1011)")
    print("\n-> results_v07final.json")

if __name__=="__main__": main()
