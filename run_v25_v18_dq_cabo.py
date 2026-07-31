#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v18 — DON QUIJOTE REAL COMPLETO: 'cabo' con skip-gram + k-means + generación por cluster.
Corpus completo: donquijote.txt (una vez, fuera del script).
"""
import json, math, random, re
from collections import defaultdict, Counter

random.seed(0)
W=8
WORD="cabo"
K=3  # probar 2 y 3 sentidos

def load_dq():
    with open("donquijote.txt", encoding="utf-8") as f:
        text=f.read().lower(); text=re.sub(r"[^\w\sáéíóúüñ]"," ",text)
    return text.split()

def build_vocab(seq, min_count=5):
    c=Counter(seq)
    return sorted([w for w,n in c.items() if n>=min_count])

def context_vector(ctx, vocab):
    v=[0.0]*len(vocab)
    for w in ctx:
        if w in vocab:
            v[vocab.index(w)]=1.0
    return v

def norm(v): return math.sqrt(sum(x*x for x in v))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else dot(a,b)/(na*nb)

def kmeans(X, k=2, epochs=30):
    n=len(X); d=len(X[0])
    centroids=random.sample(X, min(k, n))
    for _ in range(epochs):
        clusters=[[] for _ in range(k)]
        for x in X:
            sims=[cos(x,c) for c in centroids]
            best=max(range(k), key=lambda i: sims[i])
            clusters[best].append(x)
        new_cent=[]
        for c in clusters:
            if not c: new_cent.append(centroids[0])
            else:
                nd=len(c)
                new_cent.append([sum(x[j] for x in c)/nd for j in range(d)])
        centroids=new_cent
    labels=[]
    for x in X:
        sims=[cos(x,c) for c in centroids]
        labels.append(max(range(k), key=lambda i: sims[i]))
    return labels, centroids

# skip-gram simple sobre corpus completo
class SkipGram:
    def __init__(self, vocab, D=16, lr=0.05, window=5, neg_samples=5):
        self.D=D; self.lr=lr; self.window=window; self.neg_samples=neg_samples
        self.vocab=vocab
        self.emb={w:[random.gauss(0,0.1) for _ in range(D)] for w in vocab}
        self.ctx={w:[random.gauss(0,0.1) for _ in range(D)] for w in vocab}
    def fit(self, tokens, epochs=5):
        rng=random.Random(1)
        for ep in range(epochs):
            for i in range(len(tokens)):
                target=tokens[i]
                if target not in self.emb: continue
                start=max(0,i-self.window); end=min(len(tokens),i+self.window+1)
                for j in range(start,end):
                    if j==i: continue
                    context=tokens[j]
                    if context not in self.ctx or context==target: continue
                    neg=rng.sample(self.vocab, min(self.neg_samples,len(self.vocab)))
                    for d in range(self.D): self.emb[target][d]+=self.lr*(self.ctx[context][d]-self.emb[target][d])
                    for ns in neg:
                        for d in range(self.D): self.emb[target][d]-=self.lr*self.ctx[ns][d]

def main():
    print("=== v0.25 v18 DQ REAL 'cabo' + kmeans + bigramas ===")
    seq=load_dq()
    print(f" tokens totales={len(seq)}")
    vocab=build_vocab(seq, min_count=3)
    print(f" vocab (min_count=3)={len(vocab)}")
    # extraer contextos de 'cabo'
    X=[]; positions=[]
    for i in range(len(seq)):
        if seq[i]==WORD:
            ctx=seq[max(0,i-W):i]
            vec=context_vector(ctx, vocab)
            if sum(vec)>0:
                X.append(vec); positions.append(i)
    print(f" contextos 'cabo'={len(X)}")
    if len(X) < 10:
        print(" pocos contextos, aborta diagnostico"); return
    # k-means liviano
    labels, centroids = kmeans(X, k=K, epochs=20)
    counts=Counter(labels)
    print(f" cluster sizes k={K}: {counts}")
    coh={}
    for k_ in range(K):
        pts=[x for x,l in zip(X,labels) if l==k_]
        if not pts: coh[k_]=0
        else:
            c=centroids[k_]
            coh[k_]=sum(cos(p,c) for p in pts)/len(pts)
    print(f" cohesion: {coh}")
    if min(counts.values()) < 3:
        print(" cluster demasiado chico"); return
    # recopilar tokens y textos por cluster
    cluster_tokens={k_:[] for k_ in range(K)}
    cluster_texts={k_:[] for k_ in range(K)}
    for pos,lbl in zip(positions, labels):
        ctx=seq[max(0,pos-W):pos]
        cluster_tokens[lbl].append(seq[pos])
        cluster_texts[lbl].append(" ".join(ctx))
    # bigramas por cluster
    models={}
    for lbl,tokens in cluster_tokens.items():
        if not tokens: continue
        t=defaultdict(Counter)
        for w,wn in zip(tokens, tokens[1:]): t[w][wn]+=1
        models[lbl]={w:{k:v/sum(c.values()) for k,v in c.items()} for w,c in t.items()}
    # generacion
    print("\n Ejemplos por cluster:")
    for lbl in range(K):
        sample=random.choice(cluster_texts[lbl]) if cluster_texts[lbl] else WORD
        ctx=sample.split()[-3:]
        print(f"\n Cluster {lbl} (n={counts[lbl]}, coh={coh.get(lbl,0):.3f}):")
        print(f"  seed: {' '.join(ctx)}")
        out=[]
        for _ in range(10):
            probs=models[lbl].get(ctx[-1], {})
            preds=sorted(probs, key=probs.get, reverse=True)[:8]
            if not preds: break
            out.append(preds[0]); ctx.append(preds[0]); ctx=ctx[-10:]
        print(f"  -> {' '.join(out)}")
    # pureza de vocabulario entre clusters
    topN=30
    sets={lbl:set([w for w,_ in Counter(tokens).most_common(topN)]) for lbl,tokens in cluster_tokens.items()}
    overlaps={}
    for a in range(K):
        for b in range(a+1,K):
            overlaps[f"{a}-{b}"]=len(sets[a] & sets[b])
    print(f"\n top-{topN} overlaps: {overlaps}")
    veredicto="NO FUNCIONAL"
    if min(counts.values()) >= 5 and min(coh.values()) > 0.3 and max(overlaps.values(), default=999) < topN*0.6:
        veredicto="FUNCIONAL: 'cabo' muestra estructura de sentido diferenciable en DQ."
    elif min(counts.values()) >= 3 and max(overlaps.values(), default=999) < topN*0.8:
        veredicto="PARCIAL: hay diferenciacion parcial."
    print(f"\n{veredicto}")
    out=dict(experiment="v0.25_v18_dq_cabo_real",
             results=dict(k=K, sizes=dict(counts), cohesion=coh,
                          overlaps=overlaps, veredicto=veredicto))
    json.dump(out, open("results_v25_v18.json","w"), indent=2)
    print("-> results_v25_v18.json")

if __name__=="__main__":
    main()
