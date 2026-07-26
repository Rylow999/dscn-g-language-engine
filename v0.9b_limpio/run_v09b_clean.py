#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.9b CORREGIDO — categorizacion SIN diccionario en el entrenamiento.
El v0.9b original usaba SUST/VERB DURANTE el train (circular). Aca: entrenamos
next-token LIMPIO (v0.6a). Luego, SOLO EN EVAL, clusterizamos el espacio omega
(k-means k=2) y medimos la pureza contra SUST/VERB. El diccionario NUNCA entra
al train. Si la pureza > azar, la geometria separa sintaxis sola.
"""
import json, math, random, re, time
from collections import Counter
D=16; V=150; BETA=0.10; EPOCHS=3; SEED=0; K=2
def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words,V): return [w for w,_ in Counter(words).most_common(V)]
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def avg_vecs(vecs):
    n=len(vecs)
    if n==0: return [0.0]*D
    return [sum(v[d] for v in vecs)/n for d in range(D)]
def kmeans(points,rng,iters=10):
    if len(points)<K: return [avg_vecs(points)]
    cents=[points[rng.randrange(len(points))] for _ in range(K)]
    for _ in range(iters):
        cl=[[] for _ in range(K)]
        for p in points:
            best=max(range(K), key=lambda c: cos(p,cents[c])); cl[best].append(p)
        for k in range(K):
            if cl[k]: cents[k]=avg_vecs(cl[k])
    # reasignar para pureza
    assign=[max(range(K), key=lambda c: cos(p,cents[c])) for p in points]
    return assign
SUST={'don','quijote','caballero','sancho','casa','gato','pez','perro','rojo','libro','dia','noche','rey','campo','espada','mujer','hombre','agua','pan','vino','tierra','cielo','sol','luna','mano','vida','muerte','amor','mundo','pueblo','castillo','senor','camino','cuerpo','historia'}
VERB={'come','corre','es','tiene','va','dice','hace','da','ve','oye','sabe','quiere','puede','debe','vive','habla','piensa','llega','pone','deja','mira','siente','ama','odia','cree','llama'}
def main():
    print("=== v0.9b CATEGORIZACION corregida (dict solo en eval) ===")
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
    pts=[omega[i] for i in range(V)]
    assign=kmeans(pts,rng)
    # pureza: para cada cluster, mayoria de clase (S/V/O)
    purity=0
    for k in range(K):
        members=[vocab[i] for i in range(V) if assign[i]==k]
        if not members: continue
        s=sum(1 for w in members if w in SUST)
        v=sum(1 for w in members if w in VERB)
        o=len(members)-s-v
        purity+=max(s,v,o)
    purity/=V
    # baseline azar: un solo cluster -> clase mayoritaria global
    S=sum(1 for w in vocab if w in SUST); Ve=sum(1 for w in vocab if w in VERB); O=V-S-Ve
    azar=max(S,Ve,O)/V
    out=dict(experiment="v0.9b_categorizacion_limpia",
             hypothesis="Tras next-token limpio, el espacio omega separa S/V por geometria (pureza de cluster > azar). Dict SOLO en eval.",
             params=dict(d=D,V=V,beta=BETA,epochs=EPOCHS,k=K),
             pureza_cluster=round(purity,4), baseline_azar=round(azar,4),
             separa=(purity>azar+0.05))
    json.dump(out,open("results_v09b_clean.json","w"),indent=2)
    print(f"pureza={purity:.4f} azar={azar:.4f}")
    print("\n-> results_v09b_clean.json")
if __name__=="__main__": main()
