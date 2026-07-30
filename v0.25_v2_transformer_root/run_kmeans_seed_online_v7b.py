#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v7b — PASO 1+2 ONLINE CON CORPUS SINTÉTICO MEJORADO CON O ENUNCIADOS CON SENTIDOS REALES (A/B) CON GROUND TRUTH por construcción.
Arregla el problema de Don Quijote: no tenía polisemia mixta utilizable. Aquí cada ocurrencia
tiene sentido A o B definido, y el corpus es mas variado que bloques de 7 palabras.
"""
import json, math, random, re
from collections import defaultdict, Counter

random.seed(0)

# ---------- basics ----------
def norm(v): return math.sqrt(sum(x*x for x in v))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else dot(a,b)/(na*nb)

# ---------- corpus mejorado: oraciones realistas con sentidos marcados ----------
POLYSEMY = {
    "banco": {
        "A": ["dinero","pagar","cuenta","ahorro","plata","banquero","interes","cheque","tarjeta","retiro"],
        "B": ["rio","agua","pez","orilla","puente","corriente","boga","remo","proa","popa"],
        "templates_A": [
            "fue al banco para {A1} y {A2} con {A3} en mano",
            "el banco aprobo el {A2} sin {A3} ni {A4} de por medio",
            "si tienes {A1} en el banco podras usar el {A2} sin {A3}",
            "cerro su {A1} en el banco despues de {A2} el {A3}",
        ],
        "templates_B": [
            "se tiro al banco del {B1} para {B2} con su {B3}",
            "el {B3} choco contra el banco de la {B1} al {B2}",
            "amarraron la barca en el banco mientras el {B1} {B2}",
            "cerca del banco se {B2} una {B3} sobre la {B1}",
        ],
    },
    "llave": {
        "A": ["puerta","cerradura","abrir","candado","cerrojo","acceso","entrar","habitacion"],
        "B": ["musica","nota","tono","cancion","melodia","acorde","partitura","instrumento"],
        "templates_A": [
            "la llave de la {A1} no {A2} el {A3} despues de {A4}",
            "perdio la llave del {A1} justo cuando iba a {A2} con {A3}",
            "la {A1} estaba cerrada y la llave {A2} el {A3} con {A4}",
        ],
        "templates_B": [
            "la llave del {B1} hizo {B2} en {B3} mientras {B4} el {B5}",
            "toco la llave en {B1} y {B2} la {B3} sin {B4}",
            "esa llave suena en {B1} como un {B2} que {B3} la {B4}",
        ],
    },
    "cabo": {
        "A": ["mar","barco","cuerda","ancla","puerto","vela","proa","marinero"],
        "B": ["ciudad","tierra","casa","calle","iglesia","pueblo","parque","plaza"],
        "templates_A": [
            "atamos el {A2} al {A1} y soltamos el {A3} al {A4}",
            "el {A1} se acerca al {A2} mientras el {A3} gira alrededor del {A4}",
            "caminamos por el {A1} y dejamos el {A2} en el {A3}",
        ],
        "templates_B": [
            "el {B2} esta al {B1} de la {B5} y cerca del {B3}",
            "visitamos el {B1} y entramos a la {B2} sin tocar la {B3}",
            "cruzamos la {B5} para llegar al {B2} del otro {B1}",
        ],
    },
}

def fill_template(tpl, sense, word):
    def rep(m):
        key=m.group(0).strip("{}")
        idx=int(key[1:])-1
        words=POLYSEMY[word][sense]
        return words[idx % len(words)]
    return re.sub(r'\{[AB]\d+\}', rep, tpl)

def build_corpus(n_per_sense=40, words=None):
    if words is None: words=list(POLYSEMY.keys())
    seq=[]; meta=[]; word_ids=[]
    word_set={w:i for i,w in enumerate(words)}
    for w in words:
        senses=POLYSEMY[w]
        for sense_label,sense_words in [("A",senses["A"]),("B",senses["B"])]:
            tpls=senses[f"templates_{sense_label}"]
            for _ in range(n_per_sense):
                tpl=random.choice(tpls)
                sentence=fill_template(tpl, sense_label, w)
                # garantizar presencia de la palabra target
                if w not in sentence.split():
                    sentence = sentence + " y " + w
                toks=sentence.split()
                for t in toks:
                    seq.append(t); meta.append(sense_label if t==w else "O"); word_ids.append(word_set.get(t,-1))
    return seq, list(dict.fromkeys(seq)), word_set, meta, word_ids

# ---------- k-means offline sobre contextos ----------
def extract_contexts(tokens, meta, word, W=10):
    contexts=[]; labels=[]
    for i,t in enumerate(tokens):
        if t==word:
            start=max(0,i-W); end=min(len(tokens),i+W+1)
            ctx=tokens[start:i]+tokens[i+1:end]
            contexts.append(ctx)
            labels.append(meta[i])
    return contexts, labels

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

def kmeans(mat, k, seed=0, max_iter=80, tol=1e-4):
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

def silhouette(mat, labels):
    n=len(mat)
    if n<=1 or len(set(labels))<=1: return 0.0
    dist=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            d=1.0-cos_sparse(mat[i],mat[j])
            dist[i][j]=d; dist[j][i]=d
    clusters=defaultdict(list)
    for i,lab in enumerate(labels): clusters[lab].append(i)
    vals=[]
    for i in range(n):
        lab=labels[i]
        same=[dist[i][j] for j in clusters[lab] if j!=i]
        a=sum(same)/len(same) if same else 0.0
        if len(clusters)>1:
            b=min((sum(dist[i][j] for j in clusters[c])/len(clusters[c]) for c in clusters if c!=lab), default=0.0)
        else:
            b=0.0
        s=(b-a)/max(a,b) if max(a,b)>1e-9 else 0.0
        vals.append(s)
    return sum(vals)/len(vals)

def cos_sparse(a,b):
    na=norm_sparse(a); nb=norm_sparse(b)
    return 0.0 if na<1e-9 or nb<1e-9 else dot_sparse(a,b)/(na*nb)
def dot_sparse(a,b): return sum(a[k]*b[k] for k in a if k in b)
def norm_sparse(v): return math.sqrt(sum(x*x for x in v.values())) if v else 0.0

# ---------- grafo online ----------
class PolysemyGraph:
    def __init__(self, D, lr=0.05, beta_anchor=0.2, beta_repulse=0.05, theta=0.8):
        self.D=D; self.lr=lr; self.beta_anchor=beta_anchor; self.beta_repulse=beta_repulse; self.theta=theta
        self.emb={}; self.sub={}
    def seed_word(self, word, seed_vecs):
        self.sub[word]=[list(seed_vecs[0]), list(seed_vecs[1])]
        self.emb[word]=[0.0]*self.D
    def update(self, word, context_words, D):
        beta=self.beta_anchor; br=self.beta_repulse; theta=self.theta
        if word not in self.sub or self.sub[word] is None: return
        A=self.sub[word][0]; B=self.sub[word][1]
        ctx=[0.0]*D; valid=0
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
            A[d]=max(-1.0,min(1.0,A[d])); B[d]=max(-1.0,min(1.0,B[d]))
        self.sub[word]=[A,B]

def run_online_seed(tokens, meta, word, seed_vecs, D=16, epochs=20, W=8):
    g=PolysemyGraph(D=D, lr=0.05, beta_anchor=0.2, beta_repulse=0.05, theta=0.8)
    g.seed_word(word, seed_vecs)
    vocab=sorted(set(tokens))
    for w in vocab:
        if w not in g.emb: g.emb[w]=[random.gauss(0,0.1) for _ in range(D)]
    dists=[]
    for ep in range(epochs):
        for i in range(W, len(tokens)):
            w=tokens[i]
            if w not in g.sub or g.sub[w] is None: continue
            ctx=tokens[max(0,i-W):i]
            g.update(w, ctx, D)
        A,B=g.sub[word]; dists.append(cos(A,B))
    return dists

# ---------- evaluacion ----------
def evaluate(ground_labels, pred_labels):
    correct=sum(1 for g,p in zip(ground_labels,pred_labels) if g==p)
    return correct/len(ground_labels)

# ---------- main ----------
def main():
    print("=== v0.25 v7b PASO 1+2 ONLINE CON CORPUS MEJORADO ===")
    words=["banco","llave","cabo"]
    seq,vocab,word_set,meta,word_ids=build_corpus(n_per_sense=60, words=words)
    print(f"seq={len(seq)} vocab={len(vocab)} words={words}")
    summary={"words":{}}
    global_results={}
    for word in words:
        contexts,gt_labels=extract_contexts(seq,meta,word,W=10)
        print(f"\n{word}: {len(contexts)} ocurrencias (A={gt_labels.count('A')}, B={gt_labels.count('B')})")
        mat,bow_vocab,bow_idx=build_bow(contexts)
        labs1,cent1,iner1=kmeans(mat,k=1,seed=0)
        labs2,cent2,iner2=kmeans(mat,k=2,seed=0)
        sil1=silhouette(mat,labs1); sil2=silhouette(mat,labs2)
        mejora=(iner1-iner2)/iner1 if iner1>1e-9 else 0.0
        print(f"  k=1 iner={iner1:.3f} sil={sil1:.3f}")
        print(f"  k=2 iner={iner2:.3f} sil={sil2:.3f} mejora={mejora:.1%}")
        if sil2>sil1+0.02 and mejora>0.05:
            verdict_kmeans="EXISTE ESTRUCTURA BIMODAL"
        elif sil2>sil1:
            verdict_kmeans="SEPARACION PARCIAL"
        else:
            verdict_kmeans="SIN SENAL CLARA"
        print(f"  kmeans: {verdict_kmeans}")
        # seed online
        D=16
        def project(c):
            v=[0.0]*D
            for pos,val in c.items(): v[pos % D]+=val
            n=norm(v)
            if n>1e-9: v=[x/n for x in v]
            return v
        seed0=project(cent2[0]); seed1=project(cent2[1])
        dists=run_online_seed(seq,meta,word,[seed0,seed1],D=D,epochs=30,W=8)
        divergence=dists[-1]-dists[0]
        if divergence < -0.05:
            verdict_online="SEPARA ONLINE desde semilla"
        elif divergence > 0.05:
            verdict_online="COLAPSA ONLINE desde semilla"
        else:
            verdict_online="ESTABLE ONLINE desde semilla"
        print(f"  online: init_cos={dists[0]:.3f} final_cos={dists[-1]:.3f} divergence={divergence:.3f} -> {verdict_online}")
        summary["words"][word]=dict(kmeans=dict(k1_inertia=iner1,k2_inertia=iner2,sil1=sil1,sil2=sil2,mejora=mejora, veredicto=verdict_kmeans), online=dict(dists=dists,init=dists[0],final=dists[-1],divergence=divergence,veredicto=verdict_online))
        global_results[word]=dict(kmeans=verdict_kmeans, online=verdict_online, mejora=mejora, sil2=sil2, init_cos=dists[0], final_cos=dists[-1])
    # cierre global
    any_separates=any(v["kmeans"]=="EXISTE ESTRUCTURA BIMODAL" and v["online"]=="SEPARA ONLINE desde semilla" for v in global_results.values())
    summary["global"]=dict(any_separates=any_separates, words=global_results)
    print("\nVEREDICTO GLOBAL:", "HAY PALABRA QUE SEPARA ONLINE CON SEMILLA" if any_separates else "NINGUNA PALABRA SEPARA CLARAMENTE CON ESTA CONFIG")
    json.dump(summary,open("results_v25_v7b.json","w"),indent=2)
    print("-> results_v25_v7b.json")
if __name__=="__main__":
    main()
