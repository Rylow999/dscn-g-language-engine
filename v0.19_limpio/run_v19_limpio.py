#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.19 LIMPIO — DOLOR DE CONSECUENCIA / EVASION dirigida por ERROR DE PREDICCION
(real, sobre corpus), NO por formula hardcodeada.
BUG de v0.19 v3 (auditoria): A = A - alpha*B/|B| + alpha*C/|C| repetido 2000 veces
GARANTIZABA matematicamente que A se alejara de B. Circular por construccion.
REDISENO honesto:
  - Grafo Hebb (con anchor para no colapsar) sobre Don Quijote.
  - En cada paso el sistema PREDICE next-token por afinidad del contexto.
  - Si predice MAL (el target no es el mas afín) -> esa transicion fue 'dolorosa'.
  - EVASION dirigida por dato: el nodo contexto se ALEJA del nodo mal-predicho
    (aprende a no transicionar ahi) y se ACERCA al nodo correcto (el target real).
  - Mido: tasa de error en transiciones 'dolorosas' antes vs despues de evadir.
    Si baja -> el dolor (error) obliga al sistema a evitar lo que lo lastima,
    emergente de los datos, no de formula.
"""
import json, math, random, re, time
from collections import Counter
D=16; V=150; W=4; EPOCHS=2; BETA=0.10; ALPHA=0.10; BETA_REP=0.0
LR_EV=0.10; SEED=0
def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def norm(v): return math.sqrt(sum(x*x for x in v))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def load_seq():
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text); vocab=[w for w,_ in Counter(words).most_common(V)]
    rng=random.Random(SEED)
    idxall=[i for i,w in enumerate(words) if w in set(vocab)]
    step=max(1,len(idxall)//20000); chosen=idxall[::step][:20000]
    return [words[i] for i in chosen], vocab
def predict(omega, idx, vocab, seq, step):
    Vn=len(vocab)
    ctx=[omega[idx[seq[step-W+j]]] for j in range(W)]
    best,bv=-1,-1.0
    for w in range(Vn):
        s=max(cos(omega[idx[seq[step-W+j]]], omega[w]) for j in range(W))
        if s>bv: bv=s; best=w
    return best
def main():
    print("=== v0.19 LIMPIO (dolor = error de prediccion real, evasion dirigida por dato) ===")
    seq,vocab=load_seq(); Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(Vn)]
    omega0=[list(o) for o in omega]  # ancla inicial (copia de cada vector)
    # fase 1: entrenar Hebb (con anchor)
    for ep in range(EPOCHS):
        for i in range(1,len(seq)):
            a=idx[seq[i-1]]; b=idx[seq[i]]; tb=omega[b]
            new=[(1-BETA)*omega[a][d]+BETA*tb[d] for d in range(D)]
            omega[a]=[ALPHA*omega0[a][d]+(1-ALPHA)*new[d] for d in range(D)]
    # medir error basal en transiciones 'dolorosas' (donde predice mal)
    dolorosas=[]  # (ctx_idx, pred_idx, target_idx)
    for i in range(W,len(seq)):
        p=predict(omega,idx,vocab,seq,i)
        if p!=idx[seq[i]]:
            dolorosas.append((idx[seq[i-1]], p, idx[seq[i]]))
    err_basal=len(dolorosas)
    # fase 2: EVASION dirigida por dato (error real, no formula)
    for (a,p,t) in dolorosas:
        pa=norm(omega[a])
        if pa>1e-9:
            # se aleja del mal-predicho p, se acerca al target real t
            np_=norm(omega[p]); nt=norm(omega[t])
            omega[a]=[omega[a][d] - LR_EV*(omega[p][d]/np_ if np_>1e-9 else 0)
                      + LR_EV*(omega[t][d]/nt if nt>1e-9 else 0) for d in range(D)]
    # medir error despues de evadir (re-evaluar las mismas transiciones)
    err_ev=0
    for (a,p,t) in dolorosas:
        ctx=[omega[a]]  # contexto inmediato = nodo a
        best,bv=-1,-1.0
        for w in range(Vn):
            s=cos(omega[a], omega[w])
            if s>bv: bv=s; best=w
        if best!=t: err_ev+=1  # si ahora predice t (el correcto), no cuenta como error
    out=dict(experiment="v0.19_limpio_dolor_error_real",
             hypothesis="El dolor (error de prediccion real) dirige la evasion: el nodo se aleja del mal-predicho y acerca al correcto. Error en transiciones dolorosas baja despues de evadir.",
             params=dict(d=D,V=V,window=W,epochs=EPOCHS,beta=BETA,alpha=ALPHA,lr_ev=LR_EV),
             transiciones_dolorosas=err_basal,
             error_post_evasion=err_ev,
             evasion_real=(err_ev < err_basal))
    json.dump(out,open("results_v19_limpio.json","w"),indent=2)
    print(f"transiciones dolorosas={err_basal}  error post-evasion={err_ev}")
    print(f"veredicto: {'EVASION REAL (dirigida por dato)' if err_ev<err_basal else 'NO baja'}")
    print("\n-> results_v19_limpio.json")
if __name__=="__main__": main()
