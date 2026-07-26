#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.17 — WSD NO SUPERVISADO + SENSE NODES sobre Don Quijote (idea 1 real).
El test de juguete (v0.15/0.15d) fallo por corpus simetrico. Aca usamos texto real
(Don Quijote) con contexto ASIMETRICO natural. Para cada palabra, clusterizamos sus
contextos (k-means k=2, Python puro). Si separan, la palabra es polisemica y se
parte en sense nodes (omega distinto por sentido, idea 1 de Luciano). Re-trenamos
con sense assignment y medimos next-token vs baseline v0.6a (10.11%).
"""
import json, math, random, re, time
from collections import Counter
D=8; BETA=0.15; SEED=0; V=150; W=2; K=2
def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words, V): return [w for w,_ in Counter(words).most_common(V)]
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def avg_vecs(vecs):
    n=len(vecs)
    if n==0: return [0.0]*D
    return [sum(v[d] for v in vecs)/n for d in range(D)]
def kmeans2(points, rng, iters=6):
    if len(points)<2: return None
    c0=points[rng.randrange(len(points))]; c1=points[rng.randrange(len(points))]
    for _ in range(iters):
        a=[p for p in points if cos(p,c0)>=cos(p,c1)]
        b=[p for p in points if p not in a]
        if not a: a=[points[0]]; b=points[1:]
        if not b: b=[points[-1]]; a=points[:-1]
        c0=avg_vecs(a); c1=avg_vecs(b)
    return c0,c1
def main():
    print("=== v0.17 WSD no supervisado + SENSE NODES (Don Quijote) ===")
    rng=random.Random(SEED)
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text); vocab=build_vocab(words,V)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    idx={w:i for i,w in enumerate(vocab)}
    seq=[w for w in words if w in set(vocab)]
    N=len(vocab)
    # 1) train base omega (single node) = v0.6a baseline
    t0=time.time()
    for i in range(1,len(seq)):
        a,b=idx[seq[i-1]],idx[seq[i]]
        omega[a]=[(1-BETA)*omega[a][k]+BETA*omega[b][k] for k in range(D)]
    print(f"base omega {time.time()-t0:.0f}s")
    # 2) collect context vectors per occurrence
    occ={w:[] for w in vocab}
    for i,w in enumerate(seq):
        ctx=[]
        for j in list(range(max(0,i-W),i))+list(range(i+1,min(len(seq),i+W+1))):
            ctx.append(omega[idx[seq[j]]])
        occ[w].append(avg_vecs(ctx))
    # 3) cluster each word -> sense nodes if separable
    senses={}  # word -> list of centroids (sense nodes)
    sep_scores={}
    n_poly=0
    for w in vocab:
        pts=occ[w]
        if len(pts)<10:  # muy rara, 1 sentido
            senses[w]=[avg_vecs(pts)]; continue
        km=kmeans2(pts,rng)
        if km is None: senses[w]=[avg_vecs(pts)]; continue
        c0,c1=km
        d=cos(c0,c1)  # negativo = opuestos = separados
        sep_scores[w]=d
        if d<0.3:  # separacion: sentidos distintos
            senses[w]=[c0,c1]; n_poly+=1
        else:
            senses[w]=[avg_vecs(pts)]
    print(f"sense discovery: {n_poly}/{N} palabras polisemicas (2 sentidos)")
    # 4) train sense-node omega: allocate slots word_s0, word_s1
    # assign each occurrence to nearest sense centroid
    nslots=N
    for w in vocab:
        if len(senses[w])>1: nslots+=1
    omega_s=[[0.0]*D for _ in range(nslots)]
    base_slot={w:idx[w] for w in vocab}
    sense_slot={}  # (w,si)->slot
    cur=V
    for w in vocab:
        if len(senses[w])>1:
            for si in range(len(senses[w])):
                sense_slot[(w,si)]=cur; omega_s[cur]=list(senses[w][si]); cur+=1
    # assign occurrence -> sense slot
    def assign(w, ctxvec):
        if w not in sense_slot: return base_slot[w]
        best=None; bd=-1e9
        for si in range(len(senses[w])):
            s=cos(ctxvec, senses[w][si])
            if s>bd: bd=s; best=sense_slot[(w,si)]
        return best
    t0=time.time()
    for i in range(1,len(seq)):
        ctx=[]
        for j in list(range(max(0,i-W),i))+list(range(i+1,min(len(seq),i+W+1))):
            ctx.append(omega[idx[seq[j]]])
        cvec=avg_vecs(ctx)
        a=assign(seq[i-1], cvec); b=idx[seq[i]]
        omega_s[a]=[(1-BETA)*omega_s[a][k]+BETA*omega[b][k] for k in range(D)]
    print(f"sense-node train {time.time()-t0:.0f}s")
    # 5) eval next-token accuracy (predict next by coseno of prev sense-node omega)
    ok=0; tot=0
    for i in range(1,len(seq)):
        cvec=avg_vecs([omega[idx[seq[j]]] for j in range(max(0,i-W),i)])
        a=assign(seq[i-1], cvec)
        if a is None: continue
        cands=sorted(((cos(omega_s[a],omega_s[idx[seq[j]]]),seq[j]) for j in range(N) if j!=idx[seq[i-1]]),reverse=True)
        if cands[0][1]==seq[i]: ok+=1
        tot+=1
    acc=ok/tot
    out=dict(experiment="v0.17_wsd_sense_nodes",
             hypothesis="WSD no supervisado sobre texto real descubre polisemia; sense nodes mejoran next-token sobre v0.6a.",
             params=dict(d=D,beta=BETA,vocab=V,window=W,k=K),
             n_polyseemous=n_poly, n_total_words=N,
             acc_sense_nodes=round(acc,4), baseline_v06a=0.1011,
             nota="Contexto asimetrico real (Don Quijote). v0.15/0.15d fallaron por corpus simetrico de juguete.")
    with open("results_v17.json","w") as f: json.dump(out,f,indent=2)
    print(f"acc sense-nodes={acc:.4f} (baseline v0.6a=0.1011) | polys={n_poly}/{N}")
    print("\n-> results_v17.json")
if __name__=="__main__": main()
