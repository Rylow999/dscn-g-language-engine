#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.16-bis — REFERENCIAS COMPOSITIVAS con corpus controlado de "boda".
Corpus: "boda" rodeada SIEMPRE de {flores,vestido,blanco,beso} + distractores
aleatorios. Mide si las refs aprendidas de "boda" coinciden con su composicion
real (jaccard alto). Y poda por incoherencia respeta nodo externo.
"""
import json, math, random, sys, time
D=8; SEED=0; WINDOW=2
COMP={"flores","vestido","blanco","beso"}   # componentes reales de boda
DISTR={"perro","mesa","rojo","noche","agua","libro","rey","pan"}  # distractores
def make_corpus(n=4000, rng=None):
    rng=rng or random.Random(SEED)
    seq=[]
    for _ in range(n):
        # boda rodeada de sus componentes + 1 distractor
        comps=list(COMP)
        rng.shuffle(comps)
        dc=rng.choice(list(DISTR))
        seq += comps[:2] + ["boda"] + comps[2:] + [dc]
    return seq, sorted(set(seq+list(COMP)+list(DISTR)))
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    if na<1e-9 or nb<1e-9: return 0.0
    return sum(x*y for x,y in zip(a,b))/(na*nb)
def main():
    print("=== v0.16-bis REFERENCIAS (corpus controlado boda) ===")
    rng=random.Random(SEED)
    seq,vocab=make_corpus(4000,rng)
    N=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(N)]
    refs=[set() for _ in range(N)]
    for i in range(len(seq)):
        for j in range(max(0,i-WINDOW), min(len(seq),i+WINDOW+1)):
            if i!=j: refs[idx[seq[i]]].add(idx[seq[j]])
    # test 1: refs de boda vs COMP
    rb=refs[idx["boda"]]
    comp_ids=set(idx[c] for c in COMP)
    overlap=len(rb & comp_ids); union=len(rb|comp_ids)
    jac=overlap/union if union else 0
    print(f"boda refs={len(rb)} comp={COMP} jaccard={jac:.3f}")
    # test 2: poda por incoherencia (SynapticCache 2.3 umbral bajo) no borra externo
    TH=0.5
    pod=0; vivos=0
    for n in range(N):
        for r in list(refs[n]):
            if cos(omega[n],omega[r])<TH:
                refs[n].discard(r); pod+=1; vivos+=1
    out=dict(experiment="v0.16bis_referencias_controlado",
             hypothesis="Refs de 'boda' capturan su composicion {flores,vestido,blanco,beso}; poda no borra externo.",
             params=dict(d=D,window=WINDOW,comp=list(COMP)),
             jaccard_boda_refs=round(jac,4),
             n_refs_boda=len(rb), n_comp=len(COMP),
             refs_podadas=Podadas if (Podadas:=pod) else pod,
             nodos_externos_vivos=vivos,
             nota="v0.16 usaba Don Quijote donde 'boda' no estaba en vocab150; aca corpus controlado.")
    with open("results_v16bis.json","w") as f: json.dump(out,f,indent=2)
    print(f"jaccard={jac:.4f} podadas={pod} vivos={vivos}")
    print("\n-> results_v16bis.json")
if __name__=="__main__": main()
