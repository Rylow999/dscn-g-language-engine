#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.6b-bis — DOLOR como Q-learning sobre aristas (no castigo post-hoc).
Dolor = error de prediccion: si el grafo elige b dado a y b NO era la palabra
real del corpus, sube penalizacion P[a][b]. La prediccion elige max(afinidad - P).
Esto CAMBIA la chance de elegir (no llega tarde como v0.6b).
"""
import json, math, random, re, sys, time
D=8; ALPHA=5.0; BETA=0.20; V=150; EPOCHS=3

def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words, V):
    from collections import Counter
    return [w for w,_ in Counter(words).most_common(V)]
def omega_of(w, rng):
    r=random.Random(hash(w)%100000); return [r.gauss(0,1) for _ in range(D)]
def norm(v): return math.sqrt(sum(x*x for x in v))
def affinity(q,w):
    d=math.sqrt(sum((a-b)**2 for a,b in zip(q,w))); return math.exp(-ALPHA*d)

def train(vocab, omega, seq, epochs):
    idx={w:i for i,w in enumerate(vocab)}
    P={}  # penalizacion de arista a->b
    for ep in range(epochs):
        for i in range(len(seq)-1):
            a=seq[i]; b=seq[i+1]
            if a not in idx or b not in idx: continue
            ia,ib=idx[a],idx[b]
            # predice siguiente usando afinidad MENOS penalizacion
            bestw,bests=-1,-1e9
            for j in range(len(omega)):
                if j==ia: continue
                pen=P.get(a,{}).get(vocab[j],0.0)
                s=affinity(omega[ia],omega[j])-pen
                if s>bests: bests=s; bestw=j
            # next-token supervisado (como v0.6a)
            omega[ia]=[(1-BETA)*omega[ia][k]+BETA*omega[ib][k] for k in range(D)]
            # DOLOR: si lo que elijo no fue la palabra real -> penalizo esa arista
            if bestw!=ib:
                P.setdefault(a,{})
                P[a][vocab[bestw]]=P[a].get(vocab[bestw],0.0)+0.1
    return P

def error_rate(vocab, omega, P, seq):
    idx={w:i for i,w in enumerate(vocab)}
    err=tot=0
    for i in range(len(seq)-1):
        a=seq[i]; b=seq[i+1]
        if a not in idx or b not in idx: continue
        ia=idx[a]; bestw,bests=-1,-1e9
        for j in range(len(omega)):
            if j==ia: continue
            pen=P.get(a,{}).get(vocab[j],0.0)
            s=affinity(omega[ia],omega[j])-pen
            if s>bests: bests=s; bestw=j
        if bestw!=idx[b]: err+=1
        tot+=1
    return err/tot if tot else 0.0

def main():
    print("=== v0.6b-bis DOLOR=error prediccion (Q-learning aristas) ===")
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text); vocab=build_vocab(words,V)
    rng=random.Random(0)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    seq=[w for w in words if w in set(vocab)]
    # error de prediccion SIN dolor (solo next-token de v0.6a)
    e0=error_rate(vocab,omega,{},seq)
    print(f"error prediccion SIN dolor: {e0:.4f}")
    t0=time.time(); P=train(vocab,omega,seq,EPOCHS); print(f"entrenado {time.time()-t0:.0f}s")
    e1=error_rate(vocab,omega,P,seq)
    print(f"error prediccion CON dolor: {e1:.4f}")
    out=dict(experiment="v0.6b_bis_qlearning",
             hypothesis="Dolor=error de prediccion (Q-learning sobre aristas) reduce error de prediccion.",
             params=dict(d=D,alpha=ALPHA,beta=BETA,V=V,epochs=EPOCHS,corpus="don_quijote"),
             error_sin_dolor=round(e0,4), error_con_dolor=round(e1,4), mejora=round(e0-e1,4))
    with open("results_v06b_bis.json","w") as f: json.dump(out,f,indent=2)
    print("\n-> results_v06b_bis.json")

if __name__=="__main__": main()
