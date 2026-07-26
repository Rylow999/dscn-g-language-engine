#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.15d — SENSE NODES (idea 1) resueltos por CONTEXTO de v0.14d.
v0.15 fallo porque next-token aplastaba los sense-omega. Aca el transformer
(v0.14d: backprop manual + head aprendido) procesa el contexto y elige a que
sense node enrutar. Si dado "fondo" el modelo activa banco_banca (no banco_silla),
la polisemia estructural queda resuelta.
"""
import json, math, random, re, time
D=16; W=4; LR=0.005; SEED=0; EPOCHS=3
BANCA=["fondo","dinero","cuenta","sucursal"]
SILLA=["madera","sentar","comoda","respaldo"]
NEUT=["es","grande","dura","cosa","otro","del","mar"]
def rnd_mat(rng): return [[rng.gauss(0,0.3) for _ in range(D)] for _ in range(D)]
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def mat_vec(M,v): return [dot(M[i],v) for i in range(len(M))]
def vec_add(a,b,a2=1.0): return [a[i]+a2*b[i] for i in range(len(a))]
def scale(v,s): return [x*s for x in v]
def softmax(l):
    mx=max(l); ex=[math.exp(x-mx) for x in l]; s=sum(ex); return [e/s for e in ex]
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def make_corpus(n,rng):
    seq=[]
    for _ in range(n):
        if rng.random()<0.5:
            fam=BANCA; sense="banco_banca"
        else:
            fam=SILLA; sense="banco_silla"
        pre=rng.choice(fam); post=rng.choice(fam)
        seq += [pre, sense, post]
        # filler para identidad de family words
        seq += [pre, rng.choice(NEUT), rng.choice(NEUT)]
    return seq
def main():
    print("=== v0.15d SENSE NODES + CONTEXTO v0.14d ===")
    rng=random.Random(SEED)
    seq=make_corpus(4000,rng)
    vocab=sorted(set(seq))
    V=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    Wq=rnd_mat(rng); Wk=rnd_mat(rng); Wv=rnd_mat(rng)
    Wo=[[rng.gauss(0,0.3) for _ in range(D)] for _ in range(V)]
    N=len(seq)
    t0=time.time()
    for ep in range(EPOCHS):
        for step in range(W,N):
            ctx=[omega[idx[seq[step-W+j]]] for j in range(W)]
            Q=[mat_vec(Wq,ctx[t]) for t in range(W)]
            K=[mat_vec(Wk,ctx[t]) for t in range(W)]
            VV=[mat_vec(Wv,ctx[t]) for t in range(W)]
            H=[None]*W; atts=[None]*W
            for t in range(W):
                sc=[dot(Q[t],K[s])/math.sqrt(D) for s in range(t+1)]
                a=softmax(sc); atts[t]=a
                h=[0.0]*D
                for s in range(t+1): h=vec_add(h,VV[s],a[s])
                H[t]=h
            hlast=H[-1]
            logits=[dot(Wo[j],hlast) for j in range(V)]
            tgt=idx[seq[step]]; pr=softmax(logits)
            d=[p for p in pr]; d[tgt]-=1.0
            for j in range(V):
                for dd in range(D): Wo[j][dd]-=LR*d[j]*hlast[dd]
            dh=[0.0]*D
            for j in range(V): dh=vec_add(dh,Wo[j],d[j])
            dWv=[[0.0]*D for _ in range(D)]; dVV=[[0.0]*D for _ in range(W)]
            aL=atts[W-1]
            datt=[dot(dh,VV[s]) for s in range(W)]
            sad=sum(aL[k]*datt[k] for k in range(W))
            ds=[aL[s]*(datt[s]-sad) for s in range(W)]
            dQ=[[0.0]*D for _ in range(W)]; dK=[[0.0]*D for _ in range(W)]
            for s in range(W):
                dQ[W-1]=vec_add(dQ[W-1],K[s],ds[s]/math.sqrt(D))
                dK[s]=vec_add(dK[s],Q[W-1],ds[s]/math.sqrt(D))
            for t in range(W):
                for i in range(D):
                    for jj in range(D): Wq[i][jj]-=LR*dQ[t][i]*ctx[t][jj]
            for s in range(W):
                for i in range(D):
                    for jj in range(D):
                        Wk[i][jj]-=LR*dK[s][i]*ctx[s][jj]; Wv[i][jj]-=LR*dVV[s][i]*ctx[s][jj]
    print(f"transformer bp {time.time()-t0:.0f}s")
    def predict_next(context_word):
        # contexto: [context_word] solito, predecir siguiente token
        ctx=[omega[idx[context_word]]]
        # rellenar ventana con padding (usamos solo 1 token real)
        while len(ctx)<W: ctx=[omega[0]]+ctx
        Q=[mat_vec(Wq,ctx[t]) for t in range(W)]
        K=[mat_vec(Wk,ctx[t]) for t in range(W)]
        VV=[mat_vec(Wv,ctx[t]) for t in range(W)]
        H=[None]*W
        for t in range(W):
            sc=[dot(Q[t],K[s])/math.sqrt(D) for s in range(t+1)]
            a=softmax(sc); h=[0.0]*D
            for s in range(t+1): h=vec_add(h,VV[s],a[s])
            H[t]=h
        logits=[dot(Wo[j],H[-1]) for j in range(V)]
        pr=softmax(logits)
        # top-1 que no sea el contexto ni neutral
        cands=sorted(((pr[j],vocab[j]) for j in range(V) if vocab[j]!=context_word),reverse=True)
        return cands[0][1]
    ok=0; tot=0
    for c in BANCA:
        if predict_next(c)=="banco_banca": ok+=1
        tot+=1
    for c in SILLA:
        if predict_next(c)=="banco_silla": ok+=1
        tot+=1
    acc_sense=ok/tot
    out=dict(experiment="v0.15d_sense_nodes_ctx",
             hypothesis="Sense nodes (idea 1) resueltos por contexto v0.14d: dado contexto, el modelo activa el sense node correcto.",
             params=dict(d=D,window=W,lr=LR,epochs=EPOCHS,vocab=V),
             acc_sense=round(acc_sense,4),
             baseline_v15=0.4987,
             nota="v0.15 fallo por next-token aplanar sentidos; v0.15d usa transformer (v0.14d) para enrutar al sense node.")
    with open("results_v15d.json","w") as f: json.dump(out,f,indent=2)
    print(f"acc_sense={acc_sense:.4f} (v0.15 daba 0.4987)")
    print("\n-> results_v15d.json")
if __name__=="__main__": main()
