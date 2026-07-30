#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v19 — DON QUIJOTE: 'tiempo' polisemia real con k-means + skip-gram acotado."""
import json, math, random, re
from collections import defaultdict, Counter

random.seed(0)
WORD="tiempo"
W=8
K=3

with open("donquijote.txt", encoding="utf-8") as f:
    text=f.read().lower(); text=re.sub(r"[^\w\sáéíóúüñ]"," ",text)
seq=text.split()
print(f"tokens={len(seq)} vocab={len(set(seq))}")
indices=[i for i,w in enumerate(seq) if w==WORD]
print(f"ocurrencias='{WORD}': {len(indices)}")
X=[]; positions=[]; ctx_words=Counter()
for idx in indices:
    ctx=seq[max(0,idx-W):idx]
    X.append(ctx); positions.append(idx); ctx_words.update(ctx)
print(f"context words unicos alrededor de '{WORD}': {len(ctx_words)}")
acotado_vocab=sorted([w for w,_ in ctx_words.most_common(400)]+[WORD])
print(f"vocab acotado={len(acotado_vocab)}")
# A) bag-of-words binario
bow=[]
for ctx in X:
    vec=[0.0]*len(acotado_vocab)
    for w in ctx:
        if w in acotado_vocab:
            vec[acotado_vocab.index(w)]=1.0
    bow.append(vec)

def norm(v): return math.sqrt(sum(x*x for x in v))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)

centroids=random.sample(bow, min(K, len(bow)))
labels=[0]*len(bow)
for it in range(25):
    clusters=[[] for _ in range(K)]
    for idx,x in enumerate(bow):
        sims=[cos(x,c) for c in centroids]
        labels[idx]=max(range(K), key=lambda i: sims[i])
        clusters[labels[idx]].append(x)
    centroids=[]
    for k_ in range(K):
        if clusters[k_]:
            centroids.append([sum(x[j] for x in clusters[k_])/len(clusters[k_]) for j in range(len(bow[0]))])
        else:
            centroids.append([0.0]*len(bow[0]))
counts=Counter(labels)
print(f"\n[A] cluster sizes k={K}: {counts}")
coh={}
for k_ in range(K):
    pts=[x for x,l in zip(bow,labels) if l==k_]
    coh[k_]=sum(cos(p,centroids[k_]) for p in pts)/len(pts) if pts else 0
print(" cohesion:", coh)
for k_ in range(K):
    idxs=[i for i,l in enumerate(labels) if l==k_][:8]
    print(f"\n Cluster {k_} (n={counts[k_]}, coh={coh[k_]:.3f}):")
    for i in idxs:
        pos=positions[i]
        ctx=seq[max(0,pos-W):pos]
        nxt=seq[pos+1:pos+4]
        print(f"  {' '.join(ctx)} [{WORD}] {' '.join(nxt)}")
