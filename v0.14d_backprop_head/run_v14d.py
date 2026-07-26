#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.14d — backprop manual CORREGIDO: head de salida APRENDIDO + lr bajo.
Bug de v0.14b/c: el head de salida era omega_base fijo (ruido del grafo) y el
modelo se estancaba en piso uniforme. v0.14d usa Wo (V x D) APRENDIDA que mapea
h_last -> logits, lr=0.005, 2 epocas. Si supera 10.11% (v0.6a), el contexto se
cierra en Python puro.
"""
import json, math, random, re, time
D=16; V=150; W=4; LR=0.005; SEED=0; CORPUS_N=20000; EPOCHS=2; BETA=0.10

def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words, V):
    from collections import Counter
    return [w for w,_ in Counter(words).most_common(V)]
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def mat_vec(M,v): return [dot(M[i],v) for i in range(len(M))]
def vec_add(a,b,a2=1.0): return [a[i]+a2*b[i] for i in range(len(a))]
def scale(v,s): return [x*s for x in v]
def softmax_logits(logits):
    mx=max(logits); ex=[math.exp(l-mx) for l in logits]; s=sum(ex); return [e/s for e in ex]

def main():
    print("=== v0.14d backprop manual CORREGIDO (head aprendido, lr=0.005) ===")
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text)[:CORPUS_N*4]; vocab=build_vocab(words,V)
    rng=random.Random(SEED)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    seq=[w for w in words if w in set(vocab)][:CORPUS_N]
    idx={w:i for i,w in enumerate(vocab)}
    t0=time.time()
    for i in range(1,len(seq)):
        a,b=seq[i-1],seq[i]; ia,ib=idx[a],idx[b]
        omega[ia]=[(1-BETA)*omega[ia][k]+BETA*omega[ib][k] for k in range(D)]
    print(f"grafo {time.time()-t0:.0f}s")
    def rnd_mat(): return [[rng.gauss(0,0.3) for _ in range(D)] for _ in range(D)]
    Wq=rnd_mat(); Wk=rnd_mat(); Wv=rnd_mat()
    # head aprendido: Wo[j][d] mapea h -> logit_j
    Wo=[[rng.gauss(0,0.3) for _ in range(D)] for _ in range(V)]
    t0=time.time()
    N=len(seq)
    for ep in range(EPOCHS):
        for step in range(W,N):
            ctx=[omega[idx[seq[step-W+j]]] for j in range(W)]
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
            logits=[dot(Wo[j],hlast) for j in range(V)]
            target=idx[seq[step]]; probs=softmax_logits(logits)
            d_logits=[p for p in probs]; d_logits[target]-=1.0
            # d_Wo
            for j in range(V):
                for d in range(D): Wo[j][d]-=LR*d_logits[j]*hlast[d]
            # d_hlast = sum_j d_logits[j] * Wo[j]
            d_hlast=[0.0]*D
            for j in range(V): d_hlast=vec_add(d_hlast,Wo[j],d_logits[j])
            d_Wv=[[0.0]*D for _ in range(D)]; d_VV=[[0.0]*D for _ in range(W)]
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
                for i in range(D):
                    for jj in range(D): Wq[i][jj]-=LR*d_Q[t][i]*ctx[t][jj]
            for s in range(W):
                for i in range(D):
                    for jj in range(D):
                        Wk[i][jj]-=LR*d_K[s][i]*ctx[s][jj]; Wv[i][jj]-=LR*d_VV[s][i]*ctx[s][jj]
        print(f"  epoca {ep+1} lista")
    print(f"transformer bp {time.time()-t0:.0f}s")
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
        logits=[dot(Wo[j],H[-1]) for j in range(V)]
        cands=sorted(((logits[j],vocab[j]) for j in range(V) if vocab[j] not in excl),reverse=True)
        return cands[0][1]
    ok=0; tot=0
    for step in range(W,N):
        if seq[step] in seq[step-W:step]: continue
        if predict(step)==seq[step]: ok+=1
        tot+=1
    acc=ok/tot
    out=dict(experiment="v0.14d_backprop_head_aprendido",
             hypothesis="Head de salida aprendido + lr bajo rompe el piso uniforme y supera v0.6a (10.11%).",
             params=dict(d=D,V=V,window=W,lr=LR,epochs=EPOCHS,corpus_n=CORPUS_N,head="aprendido Wo"),
             acc_hibrido=round(acc,4), baseline_v06a=0.1011,
             nota="Corrige bug de v0.14b/c: head fijo=omega_base lo estancaba. Ahora Wo aprende.")
    with open("results_v14d.json","w") as f: json.dump(out,f,indent=2)
    print(f"acc hibrido d={acc:.4f}  (baseline v0.6a=0.1011)")
    print("\n-> results_v14d.json")

if __name__=="__main__": main()
