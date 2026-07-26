#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.9c CORREGIDO — dolor = error de prediccion (senal REAL, no reward fijo).
El v0.9c original subia G por un reward constante (circular). Aca: el DOLOR es
el error de next-token (1 - P(correcto)). Dos sistemas sobre Don Quijote:
  A: omega fijo (NO aprende) -> error queda alto (no sobrevive).
  B: aprende omega por next-token (v0.6a) -> error baja (el dolor lo hace cambiar).
Medimos el error de prediccion epoca a epoca. Si B baja y A no, el "dolor interno"
(del dato) obliga al cambio que mejora la supervivencia. Sin omega_ideal trucho.
"""
import json, math, random, re, time
from collections import Counter
D=16; V=150; BETA=0.10; EPOCHS=4; SEED=0
def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words,V): return [w for w,_ in Counter(words).most_common(V)]
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def loss_nt(vocab,omega,seq,n=3000):
    idx={w:i for i,w in enumerate(vocab)}; Vn=len(vocab)
    ok=tot=0
    for i in range(1,min(n,len(seq))):
        if seq[i-1] not in idx or seq[i] not in idx: continue
        q=omega[idx[seq[i-1]]]; excl={seq[i-1]}; best,bv=-1,-1.0
        for j,o in enumerate(omega):
            if vocab[j] in excl: continue
            s=cos(q,o)
            if s>bv: bv=s; best=j
        if vocab[best]==seq[i]: ok+=1
        tot+=1
    return 1.0-(ok/tot) if tot else 1.0
def main():
    print("=== v0.9c DOLOR corregido (error de prediccion real) ===")
    rng=random.Random(SEED)
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text); vocab=build_vocab(words,V)
    seq=[w for w in words if w in set(vocab)]
    idx={w:i for i,w in enumerate(vocab)}
    # A: fijo
    omegaA=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    # B: aprende
    omegaB=[list(o) for o in omegaA]
    errA=[]; errB=[]
    for ep in range(EPOCHS):
        errA.append(round(loss_nt(vocab,omegaA,seq),4))
        errB.append(round(loss_nt(vocab,omegaB,seq),4))
        for i in range(1,len(seq)):
            a,b=idx[seq[i-1]],idx[seq[i]]
            omegaB[a]=[(1-BETA)*omegaB[a][k]+BETA*omegaB[b][k] for k in range(D)]
    print(f"A(fijo) err={errA}")
    print(f"B(aprende) err={errB}")
    out=dict(experiment="v0.9c_dolor_error_real",
             hypothesis="El dolor (error de prediccion del dato) obliga al cambio: B aprende y baja error; A fijo no. Sin reward constante.",
             params=dict(d=D,V=V,beta=BETA,epochs=EPOCHS),
             error_fijo_A=errA, error_aprende_B=errB,
             dolor_funciona=(errB[-1]<errB[0]-0.02 and errA[-1]>errA[0]-0.01))
    json.dump(out,open("results_v09c_clean.json","w"),indent=2)
    print("\n-> results_v09c_clean.json")
if __name__=="__main__": main()
