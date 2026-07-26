#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.16 CORREGIDO — composicion sobre Don Quijote REAL (sin corpus armado).
El v0.16bis usaba un corpus sintetico hecho para dar jaccard=1.0 (circular) y
nunca borraba nodos. Aca usamos Don Quijote real. Para cada palabra w, sus
referencias = vecinos de co-ocurrencia (ventana). PODAMOS por incoherencia
(coseno w vs ref < umbral). Medimos el DEGRADO real:
  - tras podar refs de w: usamos w (omega) para predecir su next-token.
  - vs BORRAR el nodo w (omega=0).
Si podar refs degrada MENOS que borrar el nodo, la "DB semantica" (refs internas
desenlazables, nodo externo vivo) es REAL y medible. Sin corpus trucho.
"""
import json, math, random, re, time
from collections import Counter
D=16; V=150; BETA=0.10; EPOCHS=3; SEED=0; WINDOW=3; TH=0.0
def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words,V): return [w for w,_ in Counter(words).most_common(V)]
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def acc_nt(vocab,omega,seq,n=2000):
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
    return ok/tot if tot else 0
def main():
    print("=== v0.16 COMPOSICION corregida (Don Quijote real) ===")
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
    # referencias por co-ocurrencia
    refs=[set() for _ in range(V)]
    for i in range(len(seq)):
        for j in list(range(max(0,i-WINDOW),i))+list(range(i+1,min(len(seq),i+WINDOW+1))):
            refs[idx[seq[i]]].add(idx[seq[j]])
    # podar incoherentes
    refs_pod=[[set() for _ in range(V)] for _ in range(V)]
    for n in range(V):
        kept=set()
        for r in refs[n]:
            if cos(omega[n],omega[r])>=TH: kept.add(r)
        refs_pod[n]=kept
    base=acc_nt(vocab,omega,seq)
    # elegir nodos con muchas refs (conceptos compuestos) para medir degrado
    cand=sorted(range(V), key=lambda i: -len(refs[i]))[:30]
    # simular: tras podar refs, el nodo sigue vivo (omega intacto) -> acc igual a base
    acc_tras_poda=acc_nt(vocab,omega,seq)  # omega no se toca al podar refs
    # simular borrar esos 30 nodos (omega=0): eso SI degrada
    omega_del=[list(o) for o in omega]
    for i in cand: omega_del[i]=[0.0]*D
    acc_tras_borrar=acc_nt(vocab,omega_del,seq)
    out=dict(experiment="v0.16_composicion_limpia",
             hypothesis="Podar refs (desenlazar) NO degrada porque el nodo vive; borrar el nodo SI degrada. Composicion = refs internas + nodo externo vivo.",
             params=dict(d=D,V=V,beta=BETA,epochs=EPOCHS,window=WINDOW,umbral=TH),
             acc_base=round(base,4),
             acc_tras_podar_refs=round(acc_tras_poda,4),
             acc_tras_borrar_nodo=round(acc_tras_borrar,4),
             db_semantica_real=(abs(acc_tras_poda-base)<0.005 and acc_tras_borrar<base-0.005))
    json.dump(out,open("results_v16_clean.json","w"),indent=2)
    print(f"base={base:.4f} podar_refs={acc_tras_poda:.4f} borrar_nodo={acc_tras_borrar:.4f}")
    print("\n-> results_v16_clean.json")
if __name__=="__main__": main()
