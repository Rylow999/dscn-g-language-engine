#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paso 1 offline: ¿existe estructura bimodal real en contextos de una palabra?
Corpus: donquijote.txt
Palabra candidata: 'banco'
Metodologia:
  - Tokenizacion basica (lowercase, sin puntuacion)
  - Para cada ocurrencia de 'banco', extraer ventana de W=10 tokens alrededor
  - Representar cada contexto como bag-of-words normalizado (tf / long_ventana)
  - KMeans offline k=2 sobre vectores de contexto
  - Comparar con k=1 (baseline)
  - Metricas: inertia, silhouette (coseno), proporcion de cluster mayoritario
 SIN mecanismo DSCN: clustering puro sobre datos reales.
"""
import json, math, random, re
from collections import defaultdict, Counter

# --- tokenizacion basica ---
def tokenize(text):
    text=text.lower()
    text=re.sub(r"[^\w\s]"," ",text)
    return text.split()

# --- cargar corpus ---
def load_corpus(path="donquijote.txt"):
    with open(path,"r",encoding="utf-8") as f:
        txt=f.read()
    toks=tokenize(txt)
    print(f"Corpus cargado: {len(toks)} tokens, {len(set(toks))} types")
    return toks

# --- extraer contextos de una palabra ---
def extract_contexts(tokens, word, W=10):
    contexts=[]; positions=[]
    for i,t in enumerate(tokens):
        if t==word:
            start=max(0,i-W); end=min(len(tokens),i+W+1)
            ctx=tokens[start:i]+tokens[i+1:end]  # excluyo la palabra misma
            contexts.append(ctx)
            positions.append(i)
    print(f"Ocurrencias de '{word}': {len(contexts)}")
    return contexts, positions

# --- vector bag-of-words normalizado ---
def build_bow(contexts):
    # vocab global de todos los contextos
    vocab=sorted(set(w for ctx in contexts for w in ctx))
    idx={w:i for i,w in enumerate(vocab)}
    print(f"Vocab contextos: {len(vocab)} types")
    # matriz: lista de vectores sparse como dicts {pos: freq}
    mat=[]
    for ctx in contexts:
        c=Counter(ctx)
        total=len(ctx) if len(ctx)>0 else 1
        vec={idx[w]:c[w]/total for w in c if w in idx}
        mat.append(vec)
    return mat, vocab, idx

# --- producto punto / norma / distancia coseno ---
def dot_sparse(a,b):
    return sum(a[k]*b[k] for k in a if k in b)
def norm_sparse(v):
    return math.sqrt(sum(x*x for x in v.values())) if v else 0.0
def cos_sparse(a,b):
    na=norm_sparse(a); nb=norm_sparse(b)
    return 0.0 if na<1e-9 or nb<1e-9 else dot_sparse(a,b)/(na*nb)

# --- k-means con distancia coseno ---
def kmeans(mat, k, seed=0, max_iter=50, tol=1e-4):
    rng=random.Random(seed)
    n=len(mat)
    if n<k: raise ValueError("menos puntos que clusters")
    # init aleatoria
    centroids=[dict(mat[i]) for i in rng.sample(range(n), k)]
    labels=[0]*n
    for it in range(max_iter):
        # asignar
        new_labels=[]
        for v in mat:
            best_j=0; best_cos=-1.0
            for j,ce in enumerate(centroids):
                c=cos_sparse(v,ce)
                if c>best_cos: best_cos=c; best_j=j
            new_labels.append(best_j)
        # recomputar centroides (promedio sparse)
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
        # convergencia
        shift=0.0
        for j in range(k):
            shift=max(shift, norm_sparse({p:abs(new_cent[j][p]-centroids[j].get(p,0.0)) for p in new_cent[j] if centroids[j].get(p,0.0)!=new_cent[j][p]}))
        centroids=[dict(c) for c in new_cent]
        labels=new_labels
        if shift<tol: break
    # inertia: suma de distancias coseno (1-cos) al centroide
    inertia=0.0
    for v,lab in zip(mat,labels):
        inertia+=1.0-cos_sparse(v,centroids[lab])
    return labels, centroids, inertia

# --- silhouette con distancia coseno ---
def silhouette(mat, labels):
    n=len(mat)
    if n<=1 or len(set(labels))<=1: return 0.0
    # precompute matriz de distancias
    dist=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            d=1.0-cos_sparse(mat[i],mat[j])
            dist[i][j]=d; dist[j][i]=d
    # por cluster: indices
    clusters=defaultdict(list)
    for i,lab in enumerate(labels): clusters[lab].append(i)
    vals=[]
    for i in range(n):
        lab=labels[i]
        same=[dist[i][j] for j in clusters[lab] if j!=i]
        a=sum(same)/len(same) if same else 0.0
        b=min((sum(dist[i][j] for j in clusters[c])/len(clusters[c]) for c in clusters if c!=lab), default=0.0)
        s=(b-a)/max(a,b) if max(a,b)>1e-9 else 0.0
        vals.append(s)
    return sum(vals)/len(vals)

def main():
    print("=== Paso 1 offline: k-means sobre contextos reales de 'banco' en Don Quijote ===")
    tokens=load_corpus()
    word="banco"
    contexts,positions=extract_contexts(tokens,word,W=10)
    if len(contexts)<4:
        print("No hay suficientes ocurrencias"); return
    mat,vocab,idx=build_bow(contexts)
    # k=1 baseline
    labs1,cent1,iner1=kmeans(mat,k=1,seed=0)
    sil1=silhouette(mat,labs1)
    # k=2
    labs2,cent2,iner2=kmeans(mat,k=2,seed=0)
    sil2=silhouette(mat,labs2)
    # proporcion del cluster mayoritario
    counts=Counter(labs2)
    maj_frac=max(counts.values())/len(labs2)
    print(f"k=1 -> inertia={iner1:.3f}, silhouette={sil1:.3f}")
    print(f"k=2 -> inertia={iner2:.3f}, silhouette={sil2:.3f}, maj_frac={maj_frac:.2f}")
    mejora_inertia=(iner1-iner2)/iner1 if iner1>1e-9 else 0.0
    print(f"mejora_inertia={mejora_inertia:.1%}")
    if sil2>sil1+0.02 and mejora_inertia>0.05:
        veredicto="EXISTE ESTRUCTURA BIMODAL: k=2 separa mejor que k=1. Vale la pena portar semilla al grafo online."
    elif sil2>sil1:
        veredicto="SEPARACION PARCIAL: k=2 mejora k=1 pero la señal es débil. Probar con otra palabra/corpus."
    else:
        veredicto="NO HAY ESTRUCTURA BIMODAL clara en contextos de 'banco' con este método. Probar otra palabra o usar reliance externa."
    print(f"VEREDICTO: {veredicto}")
    out=dict(experiment="kmeans_offline_banco_donquijote",
             hypothesis="Si k=2 da mejor silhouette/inertia que k=1 sobre contextos bag-of-words reales, existe señal bimodal en el corpus.",
             params=dict(word=word,W=10,k=2,seed=0,n_contexts=len(contexts),n_vocab=len(vocab)),
             results=dict(k1_inertia=iner1,k1_silhouette=sil1,k2_inertia=iner2,k2_silhouette=sil2,mejora_inertia=mejora_inertia,maj_frac=maj_frac,veredicto=veredicto))
    json.dump(out,open("results_kmeans_banco.json","w"),indent=2)
    print("-> results_kmeans_banco.json")
if __name__=="__main__":
    main()
