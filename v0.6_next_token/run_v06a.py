#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.6a — next-token learning sobre corpus REAL (Don Quijote).
Estilo GPT-1 pero en grafo LOCAL: cada palabra=un nodo con omega.
Regla: dada palabra w_i, el omega de w_i aprende acercarse al omega de w_{i+1}
(objetivo = palabra real, NO omega_ideal fijo). Mide accuracy next-token.
"""
import json, math, random, re, sys, time
D=8; ALPHA=5.0; BETA=0.20; V=200; EPOCHS=3

def tok(text):
    words=re.findall(r"[a-záéíóúñü]+", text.lower())
    return words

def build_vocab(words, V):
    from collections import Counter
    c=Counter(words); vocab=[w for w,_ in c.most_common(V)]
    return vocab

def omega_of(w, rng):
    r=random.Random(hash(w)%100000)
    return [r.gauss(0,1) for _ in range(D)]
def norm(v): return math.sqrt(sum(x*x for x in v))
def affinity(q,w):
    d=math.sqrt(sum((a-b)**2 for a,b in zip(q,w)))
    return math.exp(-ALPHA*d)

def train(vocab, omega, seq, epochs):
    idx={w:i for i,w in enumerate(vocab)}
    for ep in range(epochs):
        for i in range(len(seq)-1):
            a=seq[i]; b=seq[i+1]
            if a in idx and b in idx:
                ia,ib=idx[a],idx[b]
                # omega[a] aprende hacia omega[b] (next-token, estilo GPT-1)
                oa=omega[ia]; ob=omega[ib]
                omega[ia]=[(1-BETA)*oa[k]+BETA*ob[k] for k in range(D)]

def predict(vocab, omega, w):
    if w not in {v:k for k,v in enumerate(vocab)}: return None
    idx={v:k for k,v in enumerate(vocab)}
    q=omega[idx[w]]
    best,bests=-1,-1.0
    for j,o in enumerate(omega):
        if j==idx[w]: continue
        s=affinity(q,o)
        if s>bests: bests=s; best=j
    return vocab[best]

def accuracy(vocab, omega, seq, n=2000):
    idx={v:k for k,v in enumerate(vocab)}
    ok=0; tot=0
    for i in range(1,min(n,len(seq)-1)):
        a=seq[i-1]; b=seq[i]
        if a in idx and b in idx:
            pred=predict(vocab,omega,a)
            if pred==b: ok+=1
            tot+=1
    return ok/tot if tot else 0.0

def main():
    print("=== v0.6a next-token sobre Don Quijote (V=%d) ===" % V)
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text)
    print("palabras totales:", len(words))
    vocab=build_vocab(words, V)
    print("vocabulario:", V, "palabras")
    rng=random.Random(0)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    seq=[w for w in words if w in set(vocab)]
    print("secuencia util:", len(seq))
    acc0=accuracy(vocab,omega,seq)
    print(f"accuracy ANTES del entrenamiento: {acc0:.4f}")
    t0=time.time()
    train(vocab,omega,seq,EPOCHS)
    print(f"entrenado en {time.time()-t0:.0f}s")
    acc1=accuracy(vocab,omega,seq)
    print(f"accuracy DESPUES: {acc1:.4f}")
    out=dict(experiment="v0.6a_next_token",
             hypothesis="El grafo aprende next-token: accuracy sube tras entrenar omega hacia palabra siguiente.",
             params=dict(d=D,alpha=ALPHA,beta=BETA,V=V,epochs=EPOCHS,corpus="don_quijote"),
             accuracy_before=round(acc0,4), accuracy_after=round(acc1,4),
             improvement=round(acc1-acc0,4))
    with open("results_v06a.json","w") as f: json.dump(out,f,indent=2)
    print("\n-> results_v06a.json")

if __name__=="__main__": main()
