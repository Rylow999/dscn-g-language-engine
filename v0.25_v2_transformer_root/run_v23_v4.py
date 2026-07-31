#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.23 v4 — HEBB 3-BODY SOBRE DON QUIJOTE.
Gap abierto: la composicion relacional 3-body (tripletas A-B-C) sigue sin
acierto real en corpus no trivial. Antecedente:
- v0.23 v3 mostró que un sustrato random aparenta aprender 3-body cuando se
  evalua con acoplado (azar 0.011, modelo 0.042). Esa "senal 4x" no aguanta
  control monosemico.
Este script intenta un experimento mas honesto:
- Usa Don Quijote real.
- Detecta TRIPLETAS VERBALES consecutivas (A B C).
- Mide co-ocurrencia versus azar.
- Evalua si la regla de composicion (A->B y B->C implica A->C) se cumple
  estadisticamente en el corpus.
"""
import math, random, re
from collections import defaultdict, Counter

random.seed(0)

# ---------- basics ----------
def norm(v): return math.sqrt(sum(x*x for x in v))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
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

# ---------- 3-body composition ----------
def extract_triples(tokens):
    triples=[]
    for i in range(len(tokens)-2):
        triples.append((tokens[i],tokens[i+1],tokens[i+2]))
    return triples

def build_vocab(triples, top_k=200):
    # tomar solo las palabras mas frecuentes para evitar ruido
    freq=Counter()
    for a,b,c in triples:
        freq[a]+=1; freq[b]+=1; freq[c]+=1
    vocab=sorted([w for w,_ in freq.most_common(top_k)])
    idx={w:i for i,w in enumerate(vocab)}
    return vocab, idx

def build_embeddings(triples, idx, D=16):
    # embeddings aleatorios iniciales
    words=sorted(idx.keys())
    emb={w:[random.gauss(0,0.1) for _ in range(D)] for w in words}
    # actualizar con Hebb 3-body: si A->B y B->C, refuerza A->C
    for a,b,c in triples:
        if a not in emb or b not in emb or c not in emb: continue
        va=emb[a]; vb=emb[b]; vc=emb[c]
        # composer AB -> C
        for d in range(D):
            va[d]+=0.01*(vb[d]*vc[d])
            vc[d]+=0.01*(va[d]*vb[d])
    # normalizar
    for w in emb:
        n=norm(emb[w])
        if n>1e-9: emb[w]=[x/n for x in emb[w]]
    return emb

def evaluate_composition(triples, emb):
    correct=0; total=0
    for a,b,c in triples:
        if a not in emb or b not in emb or c not in emb: continue
        va=emb[a]; vb=emb[b]; vc=emb[c]
        # prediccion: compuesto AB deberia parecerse a C
        pred=[va[d]+vb[d] for d in range(len(va))]
        n=norm(pred)
        if n>1e-9: pred=[x/n for x in pred]
        if cos(pred,vc) > 0.5:
            correct+=1
        total+=1
    return correct/total if total>0 else 0.0

# ---------- baseline aleatorio ----------
def random_baseline(triples, words, D=16):
    rnd={w:[random.gauss(0,0.1) for _ in range(D)] for w in words}
    for w in rnd:
        n=norm(rnd[w])
        if n>1e-9: rnd[w]=[x/n for x in rnd[w]]
    return evaluate_composition(triples, rnd)

def main():
    print("=== v0.23 v4 HEBB 3-BODY SOBRE DON QUIJOTE ===")
    tokens=load_corpus()
    triples=extract_triples(tokens)
    vocab,idx=build_vocab(triples, top_k=200)
    print(f"triples={len(triples)} vocab={len(vocab)}")
    triples_in=[t for t in triples if t[0] in idx and t[1] in idx and t[2] in idx]
    print(f"triples in vocab={len(triples_in)}")
    emb=build_embeddings(triples_in, idx, D=16)
    acc=evaluate_composition(triples_in, emb)
    acc_r=random_baseline(triples_in, vocab, D=16)
    print(f"acc_comp={acc:.3f} acc_random={acc_r:.3f} diff={acc-acc_r:.3f}")
    out=dict(experiment="v0.23_v4_hebb_3body", params=dict(D=16, top_k=200, corpus="donquijote.txt"),
             results=dict(total_triples=len(triples), in_vocab=len(triples_in),
                          acc_composition=acc, acc_random=acc_r, diff=acc-acc_r))
    import json
    json.dump(out,open("results_v23_v4.json","w"),indent=2)
    print("-> results_v23_v4.json")
if __name__=="__main__":
    main()
