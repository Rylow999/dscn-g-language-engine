#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v17 — CORPUS REAL: Don Quijote, sentido 'vino' via k-means + bigramas por cluster.
Sin ground truth. Usamos clusters como pseudo-sentidos A/B.
"""
import json, math, random, re
from collections import defaultdict, Counter

random.seed(0)
W=8

def load_text():
    with open("donquijote.txt", encoding="utf-8") as f:
        text=f.read()
    text=text.lower(); text=re.sub(r"[^\w\sáéíóúüñ]"," ",text)
    return text.split()

def context_vectors(seq, word, vocab, W=8):
    X=[]; positions=[]
    for i in range(len(seq)):
        if seq[i]!=word:
            continue
        ctx=seq[max(0,i-W):i]
        vec=[0.0]*len(vocab)
        for w in ctx:
            if w in vocab:
                vec[vocab.index(w)]=1.0
        if sum(vec)>0:
            X.append(vec); positions.append(i)
    return X, positions

def dot(a,b): return sum(x*y for x,y in zip(a,b))
def norm(v): return math.sqrt(sum(x*x for x in v))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else dot(a,b)/(na*nb)

def kmeans(X, k=2, epochs=20):
    n=len(X); d=len(X[0])
    centroids=random.sample(X, k)
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

def main():
    print("=== v0.25 v17 DON QUIJOTE REAL: 'vino' ===")
    seq=load_text()
    print(f" tokens totales={len(seq)}")
    word="vino"
    vocab=sorted(set(seq))
    X, positions = context_vectors(seq, word, vocab, W=W)
    print(f" contextos 'vino'={len(X)} vocab_size={len(vocab)}")
    if len(X)<10:
        print(" muy pocos contextos"); return
    labels, centroids = kmeans(X, k=2, epochs=20)
    counts=Counter(labels)
    print(f" cluster sizes: {counts}")
    # cohesion dentro de cada cluster
    coh={}
    for k_ in [0,1]:
        pts=[x for x,l in zip(X,labels) if l==k_]
        if not pts: coh[k_]=0
        else:
            c=centroids[k_]
            coh[k_]=sum(cos(p,c) for p in pts)/len(pts)
    print(f" cohesion: {coh}")
    # extraer contexto parcial como texto
    A_texts=[]; B_texts=[]
    for pos,lbl in zip(positions, labels):
        ctx=seq[max(0,pos-W):pos]
        txt=" ".join(ctx)
        if lbl==0: A_texts.append(txt)
        else: B_texts.append(txt)
    print(f"\n Ejemplos A (len={len(A_texts)}):")
    for t in A_texts[:5]: print(f"  {t} ...")
    print(f"\n Ejemplos B (len={len(B_texts)}):")
    for t in B_texts[:5]: print(f"  {t} ...")
    # entrenar bigramas por cluster
    sense_seq={0:[],1:[]}
    for pos,lbl in zip(positions, labels):
        sense_seq[lbl].append(seq[pos])
    models={}
    for lbl,tokens in sense_seq.items():
        t=defaultdict(Counter)
        for w,wn in zip(tokens, tokens[1:]): t[w][wn]+=1
        models[lbl]={w:{k:v/sum(c.values()) for k,v in c.items()} for w,c in t.items()}
    # generar desde una semilla real de cada cluster
    seeds={"A": random.choice(A_texts).split()[-3:] if A_texts else ["el","vino"], "B": random.choice(B_texts).split()[-3:] if B_texts else ["el","vino"]}
    print("\n Generacion por cluster:")
    for sense_name in ["A", "B"]:
        ctx=seeds[sense_name]
        lbl=0 if sense_name=="A" else 1
        out=[]
        for _ in range(12):
            probs=models[lbl].get(ctx[-1], {})
            preds=sorted(probs, key=probs.get, reverse=True)[:8]
            if not preds: break
            out.append(preds[0]); ctx.append(preds[0]); ctx=ctx[-10:]
        print(f"  {' '.join(seeds[sense_name])} -> {' '.join(out)}")
    # pureza: usamos tokens característicos como proxy de sentido
    tokens_A=Counter([t for t in sense_seq[0]])
    tokens_B=Counter([t for t in sense_seq[1]])
    topA=set([w for w,_ in tokens_A.most_common(20)])
    topB=set([w for w,_ in tokens_B.most_common(20)])
    purity_A=sum(tokens_A.get(w,0) for w in topB if w not in topA)/sum(tokens_A.values()) if sum(tokens_A.values()) else 0
    purity_B=sum(tokens_B.get(w,0) for w in topA if w not in topB)/sum(tokens_B.values()) if sum(tokens_B.values()) else 0
    # métrica útil: diferenciación entre clusters
    overlap = len(topA & topB)
    diff = len(topA ^ topB)
    print(f"\n overlap={overlap}, diff={diff}")
    veredicto="NO FUNCIONAL"
    if diff>overlap*2 and len(A_texts)>5 and len(B_texts)>5:
        veredicto="FUNCIONAL: clusters separan vocabulario de 'vino' en Don Quijote."
    elif overlap < diff:
        veredicto="PARCIAL: hay diferenciacion parcial."
    print(f"\n{veredicto}")
    out=dict(experiment="v0.25_v17_donquijote_vino_real",
             results=dict(sizes=dict(A=len(A_texts),B=len(B_texts)),
                          cohesion=coh, veredicto=veredicto,
                          topA=list(topA)[:15], topB=list(topB)[:15],
                          overlap=overlap, diff=diff))
    json.dump(out, open("results_v25_v17.json","w"), indent=2)
    print("-> results_v25_v17.json")

if __name__=="__main__":
    main()
