#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.3b v3 / v0.16 v3 — MEMORIA y COMPOSICION con metrica DESAGREGADA.
El bug persistia: 'borrar' (omega=0) no degradaba porque el predict por coseno
salta vectores nulos (les da coseno 0, nunca se eligen) -> accuracy sobre el resto
subia. Metrica correcta: desagregar accuracy por si el TARGET estaba en el conjunto
borrado. Si borre esos nodos, el sistema NO puede acertarlos -> acc(targets_borrados)~0.
Si hiberne (viven), acc(targets_borrados)~base. Eso captura 'perder acceso' de verdad.
"""
import json, math, random, re, time
from collections import Counter
D=16; V=150; BETA=0.10; EPOCHS=3; SEED=0; WINDOW=3
def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words,V): return [w for w,_ in Counter(words).most_common(V)]
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def acc_on_targets(vocab,omega,seq,bad_targets):
    """accuracy SOLO sobre targets que estan en bad_targets (los borrados)."""
    idx={w:i for i,w in enumerate(vocab)}
    ok=tot=0
    for i in range(1,len(seq)):
        if seq[i-1] not in idx or seq[i] not in idx: continue
        if seq[i] not in bad_targets: continue   # solo targets borrados
        q=omega[idx[seq[i-1]]]; excl={seq[i-1]}; best,bv=-1,-1.0
        for j,o in enumerate(omega):
            if vocab[j] in excl: continue
            s=cos(q,o)
            if s>bv: bv=s; best=j
        if vocab[best]==seq[i]: ok+=1
        tot+=1
    return ok/tot if tot else 0.0
def main():
    print("=== v0.3b v3 / v0.16 v3 MEMORIA+COMPOSICION (metrica desagregada) ===")
    rng=random.Random(SEED)
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text); vocab=build_vocab(words,V)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    seq=[w for w in words if w in set(vocab)]
    idx={w:i for i,w in enumerate(vocab)}
    t0=time.time()
    for ep in range(EPOCHS):
        for i in range(1,len(seq)):
            a,b=idx[seq[i-1]],idx[seq[i]]
            omega[a]=[(1-BETA)*omega[a][k]+BETA*omega[b][k] for k in range(D)]
    print(f"train {time.time()-t0:.0f}s")
    # nodos a "borrar": los 30 con mas co-ocurrencia (conceptos compuestos)
    refs=[set() for _ in range(V)]
    for i in range(len(seq)):
        for j in list(range(max(0,i-WINDOW),i))+list(range(i+1,min(len(seq),i+WINDOW+1))):
            refs[idx[seq[i]]].add(idx[seq[j]])
    cand=sorted(range(V), key=lambda i: -len(refs[i]))[:30]
    bad=set(vocab[i] for i in cand)
    # A: hiberna (omega vive)
    omega_h=[list(o) for o in omega]
    acc_h=acc_on_targets(vocab,omega_h,seq,bad)
    # B: borra (omega=0)
    omega_d=[list(o) for o in omega]
    for i in cand: omega_d[i]=[0.0]*D
    acc_d=acc_on_targets(vocab,omega_d,seq,bad)
    out=dict(experiment="v0.3b_v3_v0.16_v3_memoria_composicion_desagregado",
             hypothesis="Sobre targets borrados: hibernar (omega vive) mantiene accuracy; borrar (omega=0) la destruye (~0). Pierde acceso de verdad.",
             params=dict(d=D,V=V,beta=BETA,epochs=EPOCHS,window=WINDOW,n_borrados=len(cand)),
             acc_targets_hibernados=round(acc_h,4),
             acc_targets_borrados=round(acc_d,4),
             memoria_composicion_real=(acc_h>0.02 and acc_d<acc_h*0.5))
    json.dump(out,open("results_v03b_v3.json","w"),indent=2)
    print(f"acc targets HIBERNADOS={acc_h:.4f} | targets BORRADOS={acc_d:.4f}")
    print("\n-> results_v03b_v3.json")
if __name__=="__main__": main()
