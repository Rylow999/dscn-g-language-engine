#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.18 REAL — transformer COMPLETO escalado (embeddings aprendidos, no grafo rústico).
v0.14d era un head aprendido SOBRE el grafo rústico (D=16, 9.6%). v0.18 es el
transformer entero con embeddings aprendidos y D mas grande, para ver si la
magnitud de next-token ESCALA. Mido accuracy igual que v0.14d (excluir ventana,
top-1) y comparo contra 9.6% (v0.14d hibrido) y 2.37% (baseline grafo solo).
"""
import json, math, random, re, time
from collections import Counter
D=32; V=150; W=4; LR=0.005; SEED=0; CORPUS_N=20000; EPOCHS=2
def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words,V):
    return [w for w,_ in Counter(words).most_common(V)]
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def mat_vec(M,v): return [dot(M[i],v) for i in range(len(M))]
def vec_add(a,b,a2=1.0): return [a[i]+a2*b[i] for i in range(len(a))]
def scale(v,s): return [x*s for x in v]
def softmax_logits(l):
    mx=max(l); ex=[math.exp(x-mx) for x in l]; s=sum(ex); return [e/s for e in ex]
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
def train_and_eval():
    seq,vocab=load_seq()
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
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
    ok=0; tot=0
    for step in range(W,N):
        if seq[step] in seq[max(0,step-W):step]: continue
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
        excl=set(seq[max(0,step-W):step])
        logits=[dot(Wo[j],H[-1]) for j in range(Vn)]
        cands=sorted(((logits[j],vocab[j]) for j in range(Vn) if vocab[j] not in excl),reverse=True)
        if cands and cands[0][1]==seq[step]: ok+=1
        tot+=1
    acc=ok/tot if tot else 0.0
    return acc
def main():
    print("=== v0.18 REAL transformer completo escalado (D=32) ===")
    acc=train_and_eval()
    out=dict(experiment="v0.18_real_transformer_completo",
             hypothesis="Transformer completo (embeddings aprendidos) con D=32 escala la magnitud de next-token sobre v0.14d (9.6% hibrido, 2.37% grafo solo).",
             params=dict(d=D,V=V,window=W,lr=LR,epochs=EPOCHS,corpus_n=CORPUS_N),
             acc_v18=round(acc,4),
             ref_v14d_hibrido=0.0958, ref_v14d_grafo_solo=0.0237,
             escala=(acc>0.0958+0.01))
    json.dump(out,open("results_v18.json","w"),indent=2)
    print(f"v0.18 acc={acc:.4f}  (v0.14d hibrido=0.0958, grafo solo=0.0237)")
    print("\n-> results_v18.json")
if __name__=="__main__": main()
