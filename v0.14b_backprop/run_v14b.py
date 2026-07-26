#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.14b HIBRIDO REAL con BACKPROP MANUAL (Python puro, sin numpy).
El grafo aporta omega_base por token (memoria/categoria/dolor). Un transformer de
1 capa con BACKPROP real (no Hebbiano) aprende contexto sobre una ventana W.
Implementamos el gradiente a mano: forward (self-attention causal + softmax +
cross-entropy) y backward (cadenas de regla sobre listas). Sin numpy/torch.
Si acc > 10.11% (v0.6a), el hibrido real funciona.
"""
import math, random, re, time
D=8; V=150; W=4; LR=0.05; SEED=0; CORPUS_N=20000; BETA=0.10

def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words, V):
    from collections import Counter
    return [w for w,_ in Counter(words).most_common(V)]
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def mat_vec(M,v): return [dot(M[i],v) for i in range(len(M))]
def vec_add(a,b,a2=1.0): return [a[i]+a2*b[i] for i in range(len(a))]
def vec_sub(a,b): return [a[i]-b[i] for i in range(len(a))]
def scale(v,s): return [x*s for x in v]
def softmax_logits(logits):
    mx=max(logits); ex=[math.exp(l-mx) for l in logits]; s=sum(ex)
    return [e/s for e in ex]

def main():
    print("=== v0.14b HIBRIDO backprop manual (transformer 1 capa) ===")
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text)[:CORPUS_N*2]; vocab=build_vocab(words,V)
    rng=random.Random(SEED)
    # grafo: omega_base
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    seq=[w for w in words if w in set(vocab)][:CORPUS_N]
    idx={w:i for i,w in enumerate(vocab)}
    t0=time.time()
    for i in range(1,len(seq)):
        a,b=seq[i-1],seq[i]; ia,ib=idx[a],idx[b]
        omega[ia]=[(1-BETA)*omega[ia][k]+BETA*omega[ib][k] for k in range(D)]
    print(f"grafo {time.time()-t0:.0f}s")
    # pesos transformer (D x D)
    def rnd_mat(): return [[rng.gauss(0,0.3) for _ in range(D)] for _ in range(D)]
    Wq=rnd_mat(); Wk=rnd_mat(); Wv=rnd_mat()
    # entrenar con backprop manual
    t0=time.time()
    for step in range(W,len(seq)):
        ctx=[omega[idx[seq[step-W+j]]] for j in range(W)]  # embeddings del grafo
        # forward: 1 capa self-attention causal
        Q=[mat_vec(Wq,ctx[t]) for t in range(W)]
        K=[mat_vec(Wk,ctx[t]) for t in range(W)]
        VV=[mat_vec(Wv,ctx[t]) for t in range(W)]
        # atencion causal: para cada t, sobre 0..t
        H=[None]*W
        atts=[None]*W
        for t in range(W):
            scores=[dot(Q[t],K[s])/math.sqrt(D) for s in range(t+1)]
            a=softmax_logits(scores)
            atts[t]=a
            h=[0.0]*D
            for s in range(t+1):
                h=vec_add(h, VV[s], a[s])
            H[t]=h
        # logits sobre vocab: logit_j = H[-1] . omega[j]
        hlast=H[-1]
        logits=[dot(hlast, omega[j]) for j in range(V)]
        # cross-entropy
        target=idx[seq[step]]
        probs=softmax_logits(logits)
        # backward
        d_logits=[p for p in probs]; d_logits[target]-=1.0  # dL/d_logit
        # d_hlast = sum_j d_logits[j] * omega[j]
        d_hlast=[0.0]*D
        for j in range(V):
            d_hlast=vec_add(d_hlast, omega[j], d_logits[j])
        # d_Wv: para cada s en 0..W-1, d_VV[s] = atts[W-1][s] * d_hlast ; d_atts
        d_Wv=[[0.0]*D for _ in range(D)]
        # solo la ultima posicion t=W-1 produce H[-1]
        a_last=atts[W-1]
        d_VV=[ [0.0]*D for _ in range(W) ]
        for s in range(W):
            d_VV[s]=scale(d_hlast, a_last[s])
        # d_atts[s] = d_hlast . VV[s]
        d_att=[dot(d_hlast, VV[s]) for s in range(W)]
        # d_scores via softmax: d_scores[s] = a_last[s]*(d_att[s] - sum_k a_last[k]*d_att[k])
        sum_ad=sum(a_last[k]*d_att[k] for k in range(W))
        d_scores=[a_last[s]*(d_att[s]-sum_ad) for s in range(W)]
        # d_Q[t], d_K[s] desde d_scores (solo t=W-1 usa Q[W-1], K[s] para s<W)
        d_Q=[[0.0]*D for _ in range(W)]
        d_K=[[0.0]*D for _ in range(W)]
        for s in range(W):
            d_Q[W-1]=vec_add(d_Q[W-1], K[s], d_scores[s]/math.sqrt(D))
            d_K[s]=vec_add(d_K[s], Q[W-1], d_scores[s]/math.sqrt(D))
        # d_Wq += d_Q[t] outer ctx[t]; d_Wk += d_K[s] outer ctx[s]; d_Wv += d_VV[s] outer ctx[s]
        for t in range(W):
            for i in range(D):
                for jj in range(D):
                    Wq[i][jj]-=LR*d_Q[t][i]*ctx[t][jj]
        for s in range(W):
            for i in range(D):
                for jj in range(D):
                    Wk[i][jj]-=LR*d_K[s][i]*ctx[s][jj]
                    Wv[i][jj]-=LR*d_VV[s][i]*ctx[s][jj]
        if (step-W)%4000==0:
            pass
    print(f"transformer bp {time.time()-t0:.0f}s")
    # evaluar
    def predict(step):
        ctx=[omega[idx[seq[step-W+j]]] for j in range(W)]
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
        excl=set(seq[step-W:step])
        cands=sorted(((dot(H[-1],omega[j]),vocab[j]) for j in range(V) if vocab[j] not in excl),reverse=True)
        return cands[0][1]
    ok=0; tot=0
    for step in range(W,len(seq)):
        if seq[step] in seq[step-W:step]: continue
        if predict(step)==seq[step]: ok+=1
        tot+=1
    acc=ok/tot
    out=dict(experiment="v0.14b_hibrido_backprop_manual",
             hypothesis="Transformer 1 capa con backprop manual sobre omega_base supera bigrama v0.6a (10.11%).",
             params=dict(d=D,V=V,window=W,lr=LR,corpus_n=CORPUS_N,aprendizaje="backprop manual (sin numpy)"),
             acc_hibrido=round(acc,4), baseline_v06a=0.1011,
             nota="Primer transformer con backprop real en Python puro. Grafo=memoria, transformer=contexto.")
    import json
    with open("results_v14b.json","w") as f: json.dump(out,f,indent=2)
    print(f"acc hibrido bp={acc:.4f}  (baseline v0.6a=0.1011)")
    print("\n-> results_v14b.json")

if __name__=="__main__": main()
