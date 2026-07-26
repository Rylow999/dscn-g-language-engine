#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.11 — ABSTRACCION: dimensiones por concepto (idea de Luciano: amor > rojo).
Cada concepto tiene rango de variacion gamma: abstracto (amor) -> gamma alto
(omega se mueve mas, mas dimensiones activas); concreto (rojo) -> gamma bajo
(omega fijo). Pregunta: dar mas libertad a lo abstracto mejora next-token y
evita colapso de representacion (concretos no pisan abstractos)?
"""
import json, math, random, re, sys, time
D=16; ALPHA=5.0; BETA=0.20; V=150; STEPS=4000; SEED=0

# conceptos abstractos vs concretos (muestra)
ABSTRACT={'amor','tiempo','vida','muerte','dios','alma','verdad','idea','mente','honor','fe','esperanza','mundo','historia','libertad','justicia'}
CONCRETE={'rojo','casa','gato','pez','perro','libro','dia','noche','rey','campo','espada','mujer','hombre','agua','pan','vino','tierra','cielo','sol','luna','mano'}

def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words, V):
    from collections import Counter
    return [w for w,_ in Counter(words).most_common(V)]
def omega_of(w, scale):
    r=random.Random(hash(w)%100000); return [r.gauss(0,scale) for _ in range(D)]
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    if na<1e-9 or nb<1e-9: return 0.0
    return sum(x*y for x,y in zip(a,b))/(na*nb)

def gamma_of(w):
    # abstracto: mas libertad (gamma alto). concreto: gamma bajo.
    return 0.3 if w in ABSTRACT else (0.1 if w in CONCRETE else 0.2)

def main():
    print("=== v0.11 ABSTRACCION: dimensiones por concepto (D=16) ===")
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text); vocab=build_vocab(words,V)
    rng=random.Random(SEED)
    # omega con escala segun gamma (abstracto mas variacion)
    omega=[[rng.gauss(0,gamma_of(w)) for _ in range(D)] for w in vocab]
    seq=[w for w in words if w in set(vocab)]
    idx={w:i for i,w in enumerate(vocab)}
    # baseline: todos gamma fijo 0.2
    omega_base=[[rng.gauss(0,0.2) for _ in range(D)] for _ in range(V)]
    def train(om):
        for i in range(1,len(seq)):
            a,b=seq[i-1],seq[i]; ia,ib=idx[a],idx[b]
            om[ia]=[(1-BETA)*om[ia][k]+BETA*om[ib][k] for k in range(D)]
        return om
    t0=time.time()
    omega=train(omega); omega_base=train(omega_base)
    print(f"entrenado {time.time()-t0:.0f}s")
    # medir: variacion de omega por tipo (abstracto debe variar mas)
    var_abs=[]; var_con=[]
    for w in vocab:
        j=idx[w]
        if w in ABSTRACT: var_abs.append(sum(x*x for x in omega[j]))
        elif w in CONCRETE: var_con.append(sum(x*x for x in omega[j]))
    spread_abs=sum(var_abs)/len(var_abs) if var_abs else 0
    spread_con=sum(var_con)/len(var_con) if var_con else 0
    # next-token accuracy
    def acc(om):
        ok=0; tot=0
        for i in range(1,len(seq)):
            a,b=seq[i-1],seq[i]
            if a not in idx or b not in idx: continue
            ia=idx[a]
            cands=sorted(((cos(om[j],om[ia]),vocab[j]) for j in range(V) if j!=ia),reverse=True)
            if cands[0][1]==b: ok+=1
            tot+=1
        return ok/tot
    a1=acc(omega); a0=acc(omega_base)
    out=dict(experiment="v0.11_abstraccion_dimensiones",
             hypothesis="Conceptos abstractos con mas libertad (gamma alto) preservan representacion y mejoran next-token.",
             params=dict(d=D,alpha=ALPHA,beta=BETA,V=V,steps=STEPS),
             spread_abstract=round(spread_abs,4), spread_concrete=round(spread_con,4),
             acc_con_gamma=round(a1,4), acc_base=round(a0,4),
             nota="Abstractos (amor,etc) gamma=0.3; concretos (rojo,etc) gamma=0.1.")
    with open("results_v11.json","w") as f: json.dump(out,f,indent=2)
    print(f"spread abstracto={spread_abs:.4f} concreto={spread_con:.4f}  acc gamma={a1:.4f} base={a0:.4f}")
    print("\n-> results_v11.json")

if __name__=="__main__": main()
