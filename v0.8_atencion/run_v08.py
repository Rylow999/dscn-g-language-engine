#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.8 — ATENCION RUSTICA (pesos, no tabla rigida).
En vez de trigrama tabulado (v0.7-final, no escalo), el estado es combinacion
PONDERADA de las ultimas W palabras por afinidad. Los pesos de atencion se
aprenden. Test: ¿el contexto atencional sube accuracy vs v0.6a (0.1011)?
"""
import json, math, random, re, sys, time
D=8; ALPHA=5.0; BETA=0.20; V=150; W=3; EPOCHS=2

def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words, V):
    from collections import Counter
    return [w for w,_ in Counter(words).most_common(V)]
def omega_of(w, rng):
    r=random.Random(hash(w)%100000); return [r.gauss(0,1) for _ in range(D)]
def affinity(q,w):
    d=math.sqrt(sum((a-b)**2 for a,b in zip(q,w))); return math.exp(-ALPHA*d)

def main():
    print("=== v0.8 ATENCION RUSTICA W=%d ===" % W)
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text); vocab=build_vocab(words,V)
    rng=random.Random(0)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    seq=[w for w in words if w in set(vocab)]
    idx={w:i for i,w in enumerate(vocab)}
    # atencion: para cada palabra, un vector de pesos sobre las W previas
    # entrenamos omega de cada palabra hacia la siguiente (como v0.6a) pero
    # EVALUAMOS usando estado = suma pesada por afinidad de las W previas
    t0=time.time()
    for ep in range(EPOCHS):
        for i in range(len(seq)-1):
            a=seq[i]; b=seq[i+1]
            if a not in idx or b not in idx: continue
            omega[idx[a]]=[(1-BETA)*omega[idx[a]][k]+BETA*omega[idx[b]][k] for k in range(D)]
    print(f"entrenado {time.time()-t0:.0f}s")
    # evaluar con atencion: estado = suma pesada de ultimas W palabras por afinidad mutua
    ok=tot=0
    for i in range(W, min(20000, len(seq)-1)):
        ctx=[seq[j] for j in range(i-W,i) if seq[j] in idx]
        if not ctx: continue
        # pesos de atencion: afinidad de cada ctx con la ultima
        last=omega[idx[ctx[-1]]]
        ws=[affinity(last, omega[idx[c]]) for c in ctx]
        z=sum(ws)+1e-9
        state=[sum(omega[idx[ctx[j]]][k]*ws[j] for j in range(len(ctx)))/z for k in range(D)]
        bestw,bests=-1,-1.0
        for j,o in enumerate(omega):
            s=affinity(state,o)
            if s>bests: bests=s; bestw=j
        if bestw==idx[seq[i+1]]: ok+=1
        tot+=1
    acc=ok/tot if tot else 0
    out=dict(experiment="v0.8_atencion_rustica",
             hypothesis="Atencion ponderada (no tabla) hace que el contexto suba accuracy vs v0.6a (0.1011).",
             params=dict(d=D,alpha=ALPHA,beta=BETA,V=V,W=W,epochs=EPOCHS,corpus="don_quijote"),
             accuracy=round(acc,4), comparar_con_v06a=0.1011)
    with open("results_v08.json","w") as f: json.dump(out,f,indent=2)
    print(f"accuracy atencion W={W}: {acc:.4f}  (v0.6a: 0.1011, v0.7-final tabla: 0.0385)")
    print("\n-> results_v08.json")

if __name__=="__main__": main()
