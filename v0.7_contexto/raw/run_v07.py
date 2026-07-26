#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.7 — VENTANA DE CONTEXTO W(t) acumulada (gap #1: el grafo solo mira la ultima palabra).
El omega de ESTADO = mezcla de las ultimas W palabras. El grafo predice la siguiente
usando el estado acumulado, no solo la palabra actual. Mide si sube accuracy next-token.
"""
import json, math, random, re, sys, time
D=8; ALPHA=5.0; BETA=0.20; V=150; W=3; EPOCHS=2

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
    print("=== v0.7 ventana de contexto W=%d ===" % W)
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text); vocab=build_vocab(words,V)
    rng=random.Random(0)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    seq=[w for w in words if w in set(vocab)]
    idx={w:i for i,w in enumerate(vocab)}
    # entrenar: estado = promedio de ultimas W palabras, objetivo = siguiente
    t0=time.time()
    for ep in range(EPOCHS):
        for i in range(W, len(seq)-1):
            ctx=seq[i-W:i]
            cidx=[idx[c] for c in ctx if c in idx]
            if not cidx: continue
            state=[sum(omega[c][k] for c in cidx)/len(cidx) for k in range(D)]
            b=seq[i+1]
            if b not in idx: continue
            ib=idx[b]
            # el estado aprende hacia la palabra siguiente
            state=[(1-BETA)*state[k]+BETA*omega[ib][k] for k in range(D)]
            # escribir el estado aprendido de vuelta en los nodos del contexto
            for c in cidx:
                omega[c]=[(1-0.1)*omega[c][k]+0.1*state[k] for k in range(D)]
    print(f"entrenado {time.time()-t0:.0f}s")
    # evaluar: dado contexto de W palabras, ¿predice la siguiente?
    ok=tot=0
    for i in range(W, min(20000, len(seq)-1)):
        ctx=seq[i-W:i]; cidx=[idx[c] for c in ctx if c in idx]
        if not cidx: continue
        state=[sum(omega[c][k] for c in cidx)/len(cidx) for k in range(D)]
        bestw,bests=-1,-1.0
        for j,o in enumerate(omega):
            s=affinity(state,o)
            if s>bests: bests=s; bestw=j
        if bestw==idx[seq[i+1]]: ok+=1
        tot+=1
    acc=ok/tot if tot else 0
    out=dict(experiment="v0.7_context_window",
             hypothesis="Ventana de contexto W aumenta accuracy next-token vs mirar solo la ultima.",
             params=dict(d=D,alpha=ALPHA,beta=BETA,V=V,W=W,epochs=EPOCHS,corpus="don_quijote"),
             accuracy=round(acc,4),
             nota="Estado=promedio de ultimas W palabras. Comparar con v0.6a (W=1, acc 0.10).")
    with open("results_v07.json","w") as f: json.dump(out,f,indent=2)
    print(f"accuracy contexto W={W}: {acc:.4f}  (v0.6a con W=1 dio 0.1011)")
    print("\n-> results_v07.json")

if __name__=="__main__": main()
