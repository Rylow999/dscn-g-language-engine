#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v20 — SUSTRATO REAL RESTRINGIDO: skip-gram solo entorno de 'tiempo'.
"""
import json, math, random, re
from collections import defaultdict, Counter

random.seed(0)
WORD="tiempo"
W=3; D=32; epochs=3; neg_samples=5

with open("donquijote.txt", encoding="utf-8") as f:
    text=f.read().lower(); text=re.sub(r"[^\w\sáéíóúüñ]"," ",text)
seq=text.split()
print("tokens=", len(seq))
indices=[i for i,w in enumerate(seq) if w==WORD]
print("ocurrencias='"+WORD+"':", len(indices))
# armar sub-corpus: todas las oraciones que contienen WORD, repetidas para aumentar señal
sentences=[]
current=[]
for tok in seq:
    current.append(tok)
    if tok in ".!?":
        if WORD in current:
            sentences.append(current)
        current=[]
if current and WORD in current:
    sentences.append(current)
print("oraciones con 'tiempo':", len(sentences))
# flatten
sub=[]
L=60  # fragmentos de ~60 tokens alrededor de cada 'tiempo'
used=set()
for idx in indices:
    start=max(0, idx-L//2)
    end=min(len(seq), idx+L//2)
    key=(start,end)
    if key in used: continue
    used.add(key)
    sub.extend(seq[start:end])
print("tokens sub-corpus=", len(sub), "fragmentos=", len(used))
if len(sub)==0:
    # fallback: usar toda la secuencia si no hay fragmentos
    sub=list(seq)
epochs=5
acotado=sorted(set(sub))
print("vocab acotado=", len(acotado))
# skip-gram
emb={w:[random.gauss(0,0.1) for _ in range(D)] for w in acotado}
ctx={w:[random.gauss(0,0.1) for _ in range(D)] for w in acotado}
rng=random.Random(1)
losses=[]
for ep in range(epochs):
    loss=0.0
    for i in range(len(sub)):
        target=sub[i]
        start=max(0,i-W); end=min(len(sub),i+W+1)
        for j in range(start,end):
            if j==i: continue
            context=sub[j]
            if context not in ctx or context==target: continue
            for d in range(D): emb[target][d]+=0.01*(ctx[context][d]-emb[target][d])
            neg=rng.sample(acotado, min(neg_samples, len(acotado)))
            for ns in neg:
                for d in range(D): emb[target][d]-=0.01*ctx[ns][d]
    losses.append(loss)
print("skip-gram entrenado.")
# vec de 'tiempo'
vt=emb.get(WORD, [0.0]*D)
print("norm(vtiempo)=", math.sqrt(sum(x*x for x in vt)))
# vecs mas cercanos
sims=[]
for w in acotado:
    if w==WORD: continue
    na=math.sqrt(sum(x*x for x in vt)); nb=math.sqrt(sum(x*x for x in emb[w]))
    sim=sum(a*b for a,b in zip(vt,emb[w]))/(na*nb+1e-9)
    sims.append((sim,w))
print("Top 15 cercanos a 'tiempo':")
for sim,w in sorted(sims, reverse=True)[:15]:
    print(f"  {sim:.3f} {w}")
# guardar solo un resumen
json.dump(dict(top15=sorted(sims, reverse=True)[:15], word=WORD), open("results_v25_v20.json","w"), indent=2)
print("-> results_v25_v20.json")
