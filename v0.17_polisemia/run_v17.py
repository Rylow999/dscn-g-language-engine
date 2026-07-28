#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.17 — POLISEMIA (idea 1 de Luciano) sobre TRANSFORMER v0.14d.
Los intentos con grafo rústico (v0.15/0.15d/0.17 grafo) fallaron porque el grafo
aplana los contextos: k-means no encontraba clusters (0/150). El transformer SÍ da
representaciones ricas (v0.14d rompio el piso). Aca usamos las salidas del
transformer como representacion de contexto y hacemos WSD no supervisado (k-means
k=2 por palabra). Si los clusters se separan, la palabra es polisemica -> sense nodes.

Verificacion de sintaxis: el script usa solo stdlib; range+list bien formado.
"""
import json, math, random, re, time
from collections import Counter
D=16; V=150; W=4; LR=0.005; SEED=0; CORPUS_N=20000; EPOCHS=2
def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words,V):
    return [w for w,_ in Counter(words).most_common(V)]
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def mat_vec(M,v): return [dot(M[i],v) for i in range(len(M))]
def vec_add(a,b,a2=1.0): return [a[i]+a2*b[i] for i in range(len(a))]
def scale(v,s): return [x*s for x in v]
def softmax_logits(l):
    mx=max(l); ex=[math.exp(x-mx) for x in l]; s=sum(ex); return [e/s for e in ex]
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def avg_vecs(vecs):
    n=len(vecs)
    if n==0: return [0.0]*D
    return [sum(v[d] for v in vecs)/n for d in range(D)]
def kmeans(points, rng, k=2, iters=10):
    if len(points)<k: return [0]*len(points), [avg_vecs(points)]
    cents=[points[rng.randrange(len(points))] for _ in range(k)]
    for _ in range(iters):
        cl=[[] for _ in range(k)]
        for p in points:
            best=max(range(k), key=lambda c: cos(p,cents[c])); cl[best].append(p)
        for kk in range(k):
            if cl[kk]: cents[kk]=avg_vecs(cl[kk])
    assign=[max(range(k), key=lambda c: cos(p,cents[c])) for p in points]
    return assign, cents
def load_seq():
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text)
    vocab=build_vocab(words,V)
    rng=random.Random(SEED)
    idxall=[i for i,w in enumerate(words) if w in set(vocab)]
    step=max(1,len(idxall)//CORPUS_N)
    chosen=idxall[::step][:CORPUS_N]
    seq=[words[i] for i in chosen]
    return seq, vocab
def main():
    print("=== v0.17 POLISEMIA (WSD no sup sobre transformer) ===")
    seq,vocab=load_seq()
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    # embeddings (capa de entrada aprendida, no grafo rústico)
    emb=[[rng.gauss(0,0.3) for _ in range(D)] for _ in range(Vn)]
    def rnd_mat(): return [[rng.gauss(0,0.3) for _ in range(D)] for _ in range(D)]
    Wq=rnd_mat(); Wk=rnd_mat(); Wv=rnd_mat()
    Wo=[[rng.gauss(0,0.3) for _ in range(D)] for _ in range(Vn)]
    N=len(seq)
    t0=time.time()
    for ep in range(EPOCHS):
        for step in range(W,N):
            ctx=[emb[idx[seq[step-W+j]]] for j in range(W)]
            Q=[mat_vec(Wq,ctx[t]) for t in range(W)]
            K=[mat_vec(Wk,ctx[t]) for t in range(W)]
            VV=[mat_vec(Wv,ctx[t]) for t in range(W)]
            H=[None]*W; atts=[None]*W
            for t in range(W):
                scores=[dot(Q[t],K[s])/math.sqrt(D) for s in range(t+1)]
                a=softmax_logits(scores); atts[t]=a
                h=[0.0]*D
                for s in range(t+1): h=vec_add(h,VV[s],a[s])
                H[t]=h
            hlast=H[-1]
            logits=[dot(Wo[j],hlast) for j in range(Vn)]
            target=idx[seq[step]]; probs=softmax_logits(logits)
            d=[p for p in probs]; d[target]-=1.0
            for j in range(Vn):
                for dd in range(D): Wo[j][dd]-=LR*d[j]*hlast[dd]
            d_hlast=[0.0]*D
            for j in range(Vn): d_hlast=vec_add(d_hlast,Wo[j],d[j])
            d_VV=[[0.0]*D for _ in range(W)]
            a_last=atts[W-1]
            for s in range(W): d_VV[s]=scale(d_hlast,a_last[s])
            d_att=[dot(d_hlast,VV[s]) for s in range(W)]
            sum_ad=sum(a_last[k]*d_att[k] for k in range(W))
            d_scores=[a_last[s]*(d_att[s]-sum_ad) for s in range(W)]
            d_Q=[[0.0]*D for _ in range(W)]; d_K=[[0.0]*D for _ in range(W)]
            for s in range(W):
                d_Q[W-1]=vec_add(d_Q[W-1],K[s],d_scores[s]/math.sqrt(D))
                d_K[s]=vec_add(d_K[s],Q[W-1],d_scores[s]/math.sqrt(D))
            # grad a emb (por la palabra en cada posicion de la ventana)
            for t in range(W):
                g=vec_add(d_Q[t],d_K[t])
                gi=vec_add(d_VV[t],g)
                wi=idx[seq[step-W+t]]
                for dd in range(D): emb[wi][dd]-=LR*gi[dd]
            for t in range(W):
                for i in range(D):
                    for jj in range(D): Wq[i][jj]-=LR*d_Q[t][i]*ctx[t][jj]
            for s in range(W):
                for i in range(D):
                    for jj in range(D):
                        Wk[i][jj]-=LR*d_K[s][i]*ctx[s][jj]; Wv[i][jj]-=LR*d_VV[s][i]*ctx[s][jj]
    print(f"transformer bp {time.time()-t0:.0f}s")
    # representacion de contexto = hlast en cada posicion donde aparece la palabra
    def context_repr(step):
        ctx=[emb[idx[seq[step-W+j]]] for j in range(W)]
        Q=[mat_vec(Wq,ctx[t]) for t in range(W)]
        K=[mat_vec(Wk,ctx[t]) for t in range(W)]
        VV=[mat_vec(Wv,ctx[t]) for t in range(W)]
        H=[None]*W
        for t in range(W):
            scores=[dot(Q[t],K[s])/math.sqrt(D) for s in range(t+1)]
            a=softmax_logits(scores)
            h=[0.0]*D
            for s in range(t+1): h=vec_add(h,VV[s],a[s])
            H[t]=h
        return H[-1]
    # WSD no supervisado: para cada palabra, clusterizar sus contextos en 2
    from collections import defaultdict
    occ=defaultdict(list)
    for step in range(W,N):
        occ[seq[step]].append(step)
    polys=0; detalles=[]
    for w in vocab:
        steps=occ[w]
        if len(steps)<8: continue   # necesita suficientes contextos
        vecs=[context_repr(s) for s in steps]
        assign,cents=kmeans(vecs,rng,k=2)
        sep=cos(cents[0],cents[1])
        # si los centroides se separan (coseno bajo) -> 2 sentidos
        if sep<0.5:
            polys+=1
            if len(detalles)<8:
                detalles.append(dict(word=w, n_ctx=len(steps), sep_coseno=round(sep,3)))
    out=dict(experiment="v0.17_polisemia_transformer",
             hypothesis="Sobre representaciones de transformer (no grafo rústico), el WSD no sup encuentra palabras con 2 sentidos separables (sense nodes).",
             params=dict(d=D,V=V,window=W,lr=LR,epochs=EPOCHS,corpus_n=CORPUS_N),
             palabras_evaluadas=V, palabras_polisemicas=polys,
             sep_threshold=0.5, ejemplos=detalles,
             veredicto=("POLISEMIA DETECTADA" if polys>0 else "no detectada"))
    json.dump(out,open("results_v17.json","w"),indent=2)
    print(f"palabras polisemicas (2 sentidos): {polys}/{V}")
    print("\n-> results_v17.json")
if __name__=="__main__": main()
