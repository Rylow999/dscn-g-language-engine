#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AUDITORIA v0.14d — baseline CORRECTO vs hibrido, MISMAS condiciones.
v0.14d (run_v14d.py) reporta 10.55% vs 10.11% (v0.6a V=200). Comparacion invalida.
Este script corre AMBOS en V=150, 20k tokens, 2 epocas, misma eval:
  - baseline: grafo solo (BETA=0.10, igual que v0.14d), predict por coseno.
  - hibrido: igual que v0.14d (transformer head aprendido, lr=0.005).
Mide la diferencia REAL. Ademas usa corpus ALEATORIO (no solo primeras 80k).
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
def softmax_logits(l):
    mx=max(l); ex=[math.exp(x-mx) for x in l]; s=sum(ex); return [e/s for e in ex]

def load_seq():
    # corpus aleatorio (no sesgado a inicio) para ser limpio
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text)
    vocab=build_vocab(words,V)
    # muestra aleatoria de posiciones para armar seq de CORPUS_N
    rng=random.Random(SEED)
    idxall=[i for i,w in enumerate(words) if w in set(vocab)]
    # tomar bloques aleatorios para preservar localidad (como texto real)
    step=max(1,len(idxall)//CORPUS_N)
    chosen=idxall[::step][:CORPUS_N]
    seq=[words[i] for i in chosen]
    return seq, vocab

def main():
    print("=== AUDITORIA v0.14d: baseline correcto vs hibrido ===")
    seq,vocab=load_seq()
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(Vn)]
    # grafo (igual BETA que v0.14d)
    t0=time.time()
    for i in range(1,len(seq)):
        a,b=idx[seq[i-1]],idx[seq[i]]
        omega[a]=[(1-BETA)*omega[a][k]+BETA*omega[b][k] for k in range(D)]
    print(f"grafo {time.time()-t0:.0f}s")

    # ---- BASELINE: grafo solo, predict por coseno (misma eval que v0.14d) ----
    def predict_grafo(step):
        q=omega[idx[seq[step-1]]]
        excl=set(seq[max(0,step-W):step])
        best,bestv=-1,-1.0
        for j,o in enumerate(omega):
            if vocab[j] in excl: continue
            s=dot(q,o)/((math.sqrt(sum(x*x for x in q))*math.sqrt(sum(x*x for x in o)))+1e-9)
            if s>bestv: bestv=s; best=j
        return vocab[best]
    ok=0; tot=0
    for step in range(W,len(seq)):
        if seq[step] in seq[max(0,step-W):step]: continue
        if predict_grafo(step)==seq[step]: ok+=1
        tot+=1
    acc_base=ok/tot
    print(f"BASELINE grafo solo (V=150): {acc_base:.4f}")

    # ---- HIBRIDO: transformer head aprendido (v0.14d) ----
    def rnd_mat(): return [[rng.gauss(0,0.3) for _ in range(D)] for _ in range(D)]
    Wq=rnd_mat(); Wk=rnd_mat(); Wv=rnd_mat()
    Wo=[[rng.gauss(0,0.3) for _ in range(D)] for _ in range(Vn)]
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
                for i in range(D):
                    for jj in range(D): Wq[i][jj]-=LR*d_Q[t][i]*ctx[t][jj]
            for s in range(W):
                for i in range(D):
                    for jj in range(D):
                        Wk[i][jj]-=LR*d_K[s][i]*ctx[s][jj]; Wv[i][jj]-=LR*d_VV[s][i]*ctx[s][jj]
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
        excl=set(seq[max(0,step-W):step])
        logits=[dot(Wo[j],H[-1]) for j in range(Vn)]
        cands=sorted(((logits[j],vocab[j]) for j in range(Vn) if vocab[j] not in excl),reverse=True)
        return cands[0][1]
    ok=0; tot=0
    for step in range(W,N):
        if seq[step] in seq[max(0,step-W):step]: continue
        if predict(step)==seq[step]: ok+=1
        tot+=1
    acc_hyb=ok/tot
    print(f"HIBRIDO (v0.14d) V=150: {acc_hyb:.4f}")
    out=dict(
        experiment="audit_v14d_baseline_correcto",
        nota="v0.14d comparaba 10.55%(V=150) vs 10.11%(V=200, otro corpus). Aca ambos en V=150, 20k tok, 2 ep, corpus aleatorio.",
        params=dict(d=D,V=V,window=W,lr=LR,epochs=EPOCHS,beta=BETA,corpus_n=CORPUS_N),
        baseline_grafo_solo=round(acc_base,4),
        hibrido_v14d=round(acc_hyb,4),
        diff_real=round(acc_hyb-acc_base,4),
        veredicto=("HIBRIDO SUPERA baseline correcto" if acc_hyb>acc_base else "NO supera")
    )
    with open("results_audit_v14d.json","w") as f: json.dump(out,f,indent=2)
    print(f"\nDIFERENCIA REAL: {acc_hyb-acc_base:+.4f}")
    print("-> results_audit_v14d.json")
if __name__=="__main__": main()
