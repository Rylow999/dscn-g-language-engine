#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.7-bis — CONTEXTO BIEN HECHO: estado c(t) SEPARADO de los omega de nodos.
No promediar y pisar omega (eso arruino v0.7). c(t) = mezcla de c(t-1) y omega actual.
El grafo predice usando c(t). Mide si sube accuracy vs v0.6a (W=1, 0.1011).
"""
import json, math, random, re, sys, time
D=8; ALPHA=5.0; BETA=0.20; V=150; W=3; EPOCHS=2; GAMMA=0.7

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
    print("=== v0.7-bis contexto SEPARADO W=%d ===" % W)
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text); vocab=build_vocab(words,V)
    rng=random.Random(0)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    seq=[w for w in words if w in set(vocab)]
    idx={w:i for i,w in enumerate(vocab)}
    c=[0.0]*D  # estado separado
    t0=time.time()
    for ep in range(EPOCHS):
        c=[0.0]*D
        for i in range(len(seq)-1):
            w=seq[i]; nxt=seq[i+1]
            if w not in idx or nxt not in idx: continue
            # actualizar estado: mezcla de estado previo y omega actual
            c=[GAMMA*c[k]+(1-GAMMA)*omega[idx[w]][k] for k in range(D)]
            # estado aprende hacia omega de la palabra REAL siguiente
            c=[(1-BETA)*c[k]+BETA*omega[idx[nxt]][k] for k in range(D)]
    print(f"entrenado {time.time()-t0:.0f}s")
    # evaluar: predecir siguiente usando estado acumulado
    ok=tot=0; c=[0.0]*D
    for i in range(min(20000, len(seq)-1)):
        w=seq[i]; nxt=seq[i+1]
        if w not in idx or nxt not in idx: continue
        c=[GAMMA*c[k]+(1-GAMMA)*omega[idx[w]][k] for k in range(D)]
        bestw,bests=-1,-1.0
        for j,o in enumerate(omega):
            s=affinity(c,o)
            if s>bests: bests=s; bestw=j
        if bestw==idx[nxt]: ok+=1
        tot+=1
    acc=ok/tot if tot else 0
    out=dict(experiment="v0.7_bis_contexto_separado",
             hypothesis="Estado c(t) separado (no promedio que pisa nodos) sube accuracy vs v0.6a (W=1).",
             params=dict(d=D,alpha=ALPHA,beta=BETA,V=V,W=W,epochs=EPOCHS,gamma=GAMMA,corpus="don_quijote"),
             accuracy=round(acc,4),
             comparar_con_v06a=0.1011)
    with open("results_v07b.json","w") as f: json.dump(out,f,indent=2)
    print(f"accuracy contexto separado W={W}: {acc:.4f}  (v0.6a W=1: 0.1011, v0.7 promedio: 0.0589)")
    print("\n-> results_v07b.json")

if __name__=="__main__": main()
