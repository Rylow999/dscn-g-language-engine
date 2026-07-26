#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.3 REV — "¿el grafo entiende?" Recuperacion de conceptos.
Valida ANTES del decoder: dado omega_query, ¿recupera el nodo correcto de la masa?
Variante A: afinidad por NORMA flotante (motor real).
Variante B: afinidad por BITS/puertas logicas (idea de Luciano: omega->bits,
            distancia por Hamming sobre los bits).
Mide accuracy top-1 para A vs B a distintos M (tamanio de vocabulario).
"""
import json, math, random, sys, time

D=8; ALPHA=5.0; SEEDS=10

def make_centroids(M, rng, spread=2.0):
    # centroides lejanos entre si para que sean distinguibles
    cents=[]
    for _ in range(M):
        v=[rng.gauss(0,spread) for _ in range(D)]
        cents.append(v)
    return cents

def to_bits(omega, bits_per_dim=8):
    # cuantiza cada dim a 'bits_per_dim' bits por umbral en la media 0
    bs=[]
    for x in omega:
        bs.append(1 if x>=0 else 0)
    # version mas rica: signo + magnitud en 2 bits (00,01,10,11)
    bs2=[]
    for x in omega:
        if x> 0.5: bs2.append((1,1))
        elif x>=0: bs2.append((1,0))
        elif x>-0.5: bs2.append((0,1))
        else: bs2.append((0,0))
    return bs2

def hamming(a,b):
    return sum(1 for x,y in zip(a,b) if x!=y)

def affinity_norm(q, w):
    d=math.sqrt(sum((a-b)**2 for a,b in zip(q,w)))
    return math.exp(-ALPHA*d)

def affinity_bits(qb, wb):
    # puertas logicas: similaridad = 1 - hamming/len ; cada par (a,b) contribuye
    L=len(qb)
    mism=hamming(qb,wb)
    return 1.0 - mism/L  # en [0,1]; mapear a exp para comparar con norma

def build_mass(M, rng):
    cents=make_centroids(M, rng)
    mass=[]  # cada nodo: (omega, bits)
    for c in cents:
        # varios nodos por concepto (ruido alrededor del centroide)
        for _ in range(3):
            w=[c[k]+rng.gauss(0,0.15) for k in range(D)]
            mass.append((w, to_bits(w)))
    return cents, mass

def query(cents, mass, rng, mode):
    # elige un concepto, hace query cerca de su centroide
    ci=rng.randrange(len(cents))
    q=[cents[ci][k]+rng.gauss(0,0.1) for k in range(D)]
    qb=to_bits(q)
    best_i,best_s=-1,-1e9
    for i,(w,wb) in enumerate(mass):
        s=affinity_norm(q,w) if mode=="norm" else affinity_bits(qb,wb)
        if s>best_s: best_s=s; best_i=i
    # ¿el nodo recuperado pertenece al concepto ci? (i//3 == ci)
    return best_i//3 == ci

def run(M, mode):
    accs=[]
    for s in range(SEEDS):
        rng=random.Random(s*7+M)
        cents, mass=build_mass(M, rng)
        ok=0
        for _ in range(20):
            if query(cents, mass, rng, mode): ok+=1
        accs.append(ok/20)
    return sum(accs)/len(accs)

def main():
    print("=== v0.3 REV: recuperacion (norm vs bits) ===")
    rows=[]
    for M in [4,16,64,256]:
        t0=time.time()
        an=run(M,"norm"); ab=run(M,"bits")
        rows.append(dict(M=M, acc_norm=round(an,3), acc_bits=round(ab,3)))
        print(f"M={M:>4} | norm={an:.3f} | bits={ab:.3f}  {time.time()-t0:.0f}s")
        sys.stdout.flush()
    out=dict(experiment="v0.3_retrieval",
             hypothesis="El grafo recupera el concepto correcto; la repr en bits conserva la semantica.",
             params=dict(d=D,alpha=ALPHA,seeds=SEEDS),
             note="Afinidad norma flotante vs bits/puertas logicas (idea Luciano).",
             results=rows)
    with open("results_v03.json","w") as f: json.dump(out,f,indent=2)
    print("\n-> results_v03.json")

if __name__=="__main__": main()
