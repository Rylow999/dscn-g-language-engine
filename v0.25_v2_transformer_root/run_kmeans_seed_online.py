#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paso 2 online: semillar omega0 del grafo con clusters k-means offline.
Paso 1 confirmo: existe estructura bimodal real en 'banco' en Don Quijote
(k=2: silhouette=0.552, mejora_inertia=57% sobre k=1, con solo 5 ocurrencias).
Este script porta esa seal al grafo online:
1) Extrae contextos reales de 'banco' en donquijote.txt
2) Corre k-means k=2 y toma los dos centros como semilla omega0 para los
   sub-nodos de 'banco' (en vez de gauss(0,1) aleatorio).
3) Corre diffusion+anchor sobre el corpus mini con ground truth.
4) Mide: si la semilla inicial separa bien, diffusion+anchor la refinan o la
   destruyen. Si no separa desde el inicio, medir si logra separar con
   repulsion sibling sobre una semilla genuina.
"""
import json, math, random, re
from collections import defaultdict, Counter

random.seed(0)

# ---------- basics ----------
def norm(v):
    return math.sqrt(sum(x*x for x in v))
def dot(a,b):
    return sum(x*y for x,y in zip(a,b))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else dot(a,b)/(na*nb)

# ---------- corpus ----------
def tokenize(text):
    text=text.lower()
    text=re.sub(r"[^\w\s]"," ",text)
    return text.split()

def load_corpus(path="donquijote.txt"):
    with open(path,"r",encoding="utf-8") as f:
        txt=f.read()
    return tokenize(txt)

def extract_contexts(tokens, word, W=10):
    contexts=[]; positions=[]
    for i,t in enumerate(tokens):
        if t==word:
            start=max(0,i-W); end=min(len(tokens),i+W+1)
            ctx=tokens[start:i]+tokens[i+1:end]
            contexts.append(ctx)
            positions.append(i)
    return contexts, positions

def build_bow(contexts):
    vocab=sorted(set(w for ctx in contexts for w in ctx))
    idx={w:i for i,w in enumerate(vocab)}
    mat=[]
    for ctx in contexts:
        c=Counter(ctx)
        total=len(ctx) if len(ctx)>0 else 1
        vec={idx[w]:c[w]/total for w in c if w in idx}
        mat.append(vec)
    return mat, vocab, idx

# ---------- kmeans ----------
def kmeans(mat, k, seed=0, max_iter=50, tol=1e-4):
    rng=random.Random(seed)
    n=len(mat)
    if n<k: raise ValueError("menos puntos que clusters")
    centroids=[dict(mat[i]) for i in rng.sample(range(n), k)]
    labels=[0]*n
    for it in range(max_iter):
        new_labels=[]
        for v in mat:
            best_j=0; best_cos=-1.0
            for j,ce in enumerate(centroids):
                c=cos_sparse(v,ce)
                if c>best_cos: best_cos=c; best_j=j
            new_labels.append(best_j)
        new_cent=[defaultdict(float) for _ in range(k)]
        cnts=[0]*k
        for lab,v in zip(new_labels,mat):
            cnts[lab]+=1
            for pos,val in v.items():
                new_cent[lab][pos]+=val
        for j in range(k):
            if cnts[j]>0:
                for pos in new_cent[j]:
                    new_cent[j][pos]/=cnts[j]
        shift=0.0
        for j in range(k):
            shift=max(shift, norm_sparse({p:abs(new_cent[j][p]-centroids[j].get(p,0.0)) for p in new_cent[j] if centroids[j].get(p,0.0)!=new_cent[j][p]}))
        centroids=[dict(c) for c in new_cent]
        labels=new_labels
        if shift<tol: break
    inertia=0.0
    for v,lab in zip(mat,labels):
        inertia+=1.0-cos_sparse(v,centroids[lab])
    return labels, centroids, inertia

def cos_sparse(a,b):
    na=norm_sparse(a); nb=norm_sparse(b)
    return 0.0 if na<1e-9 or nb<1e-9 else dot_sparse(a,b)/(na*nb)
def dot_sparse(a,b):
    return sum(a[k]*b[k] for k in a if k in b)
def norm_sparse(v):
    return math.sqrt(sum(x*x for x in v.values())) if v else 0.0

# ---------- grafo online ----------
class PolysemyGraph:
    def __init__(self, D, lr=0.05, beta_anchor=0.2, beta_repulse=0.05, theta=0.8):
        self.D=D; self.lr=lr; self.beta_anchor=beta_anchor
        self.beta_repulse=beta_repulse; self.theta=theta
        self.emb={}; self.sub={}  # word -> [vecA, vecB] or None

    def seed_word(self, word, seed_vecs):
        """Inicializa omega0 del grafo desde semilla externa (ej: kmeans centros)."""
        dense=[]
        for sv in seed_vecs:
            if hasattr(sv, "items"):
                dense.append(self._to_dense(sv))
            else:
                dense.append([0.0]*self.D if len(sv)<self.D else list(sv)[:self.D])
        self.sub[word]=[list(dense[0]), list(dense[1])]
        self.emb[word]=[0.0]*self.D

    def _to_dense(self, sparse_vec):
        v=[0.0]*self.D
        for pos,val in sparse_vec.items():
            if 0<=pos<self.D: v[pos]=val
        n=norm(v)
        if n>1e-9: v=[x/n for x in v]
        return v

    def update(self, word, context_words, D):
        beta=self.beta_anchor; br=self.beta_repulse; theta=self.theta
        if word not in self.sub or self.sub[word] is None: return
        A=self.sub[word][0]; B=self.sub[word][1]
        ctx=[0.0]*D
        valid=0
        for w in context_words:
            if w in self.emb:
                for d in range(D): ctx[d]+=self.emb[w][d]
                valid+=1
        if valid>0: ctx=[x/valid for x in ctx]
        ca=cos(ctx,A); cb=cos(ctx,B)
        if ca>=cb:
            for d in range(D): A[d]+=beta*(ctx[d]-A[d])
            if ca<br or ca<theta:
                for d in range(D): B[d]-=br*A[d]
        else:
            for d in range(D): B[d]+=beta*(ctx[d]-B[d])
            if cb<br or cb<theta:
                for d in range(D): A[d]-=br*B[d]
        for d in range(D):
            A[d]=max(-1.0,min(1.0,A[d]))
            B[d]=max(-1.0,min(1.0,B[d]))
        self.sub[word]=[A,B]

# ---------- experimento online ----------
def run_online_seed(tokens, word, seed_vecs, D=16, epochs=5, W=8):
    """Corre diffusion+anchor sobre 'word' usando seed_vecs como omega0.
    Mide: ¿la distancia entre sub-nodos aumenta o disminuye con el entrenamiento?
    Si aumenta: diffusion+anchor refinan la hipótesis inicial.
    Si disminuye: colapsan (no separan).
    Si se mantienen: están estables."""
    g=PolysemyGraph(D=D, lr=0.05, beta_anchor=0.2, beta_repulse=0.05, theta=0.8)
    g.seed_word(word, seed_vecs)
    vocab=sorted(set(tokens))
    for w in vocab:
        if w not in g.emb:
            g.emb[w]=[random.gauss(0,0.1) for _ in range(D)]
    dists=[]
    for ep in range(epochs):
        for i in range(W, len(tokens)):
            w=tokens[i]
            if w not in g.sub or g.sub[w] is None: continue
            ctx_words=tokens[max(0,i-W):i]
            g.update(w, ctx_words, D)
        A,B=g.sub[word]
        dists.append(cos(A,B))
    return dists

# ---------- main ----------
def main():
    print("=== Paso 2 online: semilla k-means en grafo DSCN-G ===")
    tokens=load_corpus()
    word="banco"
    contexts,positions=extract_contexts(tokens,word,W=10)
    print(f"Ocurrencias de '{word}': {len(contexts)}")
    mat,vocab,idx=build_bow(contexts)
    # k-means k=2
    labs,centers,iner=kmeans(mat,k=2,seed=0)
    print(f"k=2 -> inertia={iner:.3f}")
    print(f"centro0 dim={len(centers[0])}, centro1 dim={len(centers[1])}")
    # Project embeddings to D=16 if necessary
    D=16
    def project(c):
        # project sparse to D-dim dense by simple hash->position mapping
        v=[0.0]*D
        for pos,val in c.items():
            d=pos % D
            v[d]+=val
        n=norm(v)
        if n>1e-9: v=[x/n for x in v]
        return v
    seed0=project(centers[0])
    seed1=project(centers[1])
    print(f"semilla cos(A,B)={cos(seed0,seed1):.3f}")
    # online
    dists=run_online_seed(tokens, word, [seed0,seed1], D=D, epochs=20, W=8)
    print("distancia cos(A,B) por epoch:", [f"{d:.3f}" for d in dists])
    init=dists[0]; final=dists[-1]
    variacion=(final-init)/init if abs(init)>1e-9 else 0.0
    if final < init - 0.05:
        veredicto="SEMILLA + ANCHOR SEPARA: los sub-nodos divergen (cos baja), la hipótesis inicial se mantiene y refina."
    elif final > init + 0.05:
        veredicto="SEMILLA + ANCHOR COLAPSA: los sub-nodos se acercan, no logra separar."
    else:
        veredicto="SEMILLA + ANCHOR ESTABLE: sin refuerzo ni colapso claro."
    print(f"VEREDICTO: {veredicto}")
    out=dict(experiment="v0.25_v7_seed_kmeans_online",
             hypothesis="Semillas k-means en omega0 mejoran la separacion online frente a aleatorios.",
             params=dict(word=word,D=D,W=8,epochs=20,lr=0.05,beta_anchor=0.2,beta_repulse=0.05,theta=0.8),
             results=dict(seed_init_cos=init,seed_final_cos=final,distances=dists,veredicto=veredicto))
    json.dump(out,open("results_v25_v7.json","w"),indent=2)
    print("-> results_v25_v7.json")
if __name__=="__main__":
    main()
