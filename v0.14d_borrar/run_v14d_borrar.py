#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.14d BORRAR — memoria/composicion sobre sustrato que PREDICE (híbrido).
El grafo rústico (v0.3b/v0.16) no mostraba "borrar destruye" porque predice ~8%.
El híbrido v0.14d predice ~9.6%, y sobre el transformer el efecto de perder un
nodo SÍ debería verse. Test:
  - base: híbrido completo.
  - B (preservar): omega de nodos top vive (incluso los "hibernamos") -> = base.
  - D (borrar): omega=0 en nodos top Y se excluyen del vocab del transformer
    (no pueden ser predichos ni predecir) -> accuracy debe BAJAR.
Si D < B por margen -> "borrar destruye" confirmado en sustrato real.
"""
import json, math, random, re, time
D=16; V=150; W=4; LR=0.005; SEED=0; CORPUS_N=20000; EPOCHS=2; BETA=0.10
def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words,V):
    from collections import Counter
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
def train_grafo(seq,vocab):
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(Vn)]
    for i in range(1,len(seq)):
        a,b=idx[seq[i-1]],idx[seq[i]]
        omega[a]=[(1-BETA)*omega[a][k]+BETA*omega[b][k] for k in range(D)]
    return omega, idx
def train_hybrid(seq,vocab,omega,idx):
    Vn=len(vocab); rng=random.Random(SEED)
    def rnd_mat(): return [[rng.gauss(0,0.3) for _ in range(D)] for _ in range(D)]
    Wq=rnd_mat(); Wk=rnd_mat(); Wv=rnd_mat()
    Wo=[[rng.gauss(0,0.3) for _ in range(D)] for _ in range(Vn)]
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
    return Wq,Wk,Wv,Wo,idx
def predict(seq,vocab,omega,idx,Wq,Wk,Wv,Wo,step,excluded):
    Vn=len(vocab)
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
    excl=set(seq[max(0,step-W):step]) | excluded
    logits=[dot(Wo[j],H[-1]) for j in range(Vn)]
    cands=sorted(((logits[j],vocab[j]) for j in range(Vn) if vocab[j] not in excl),reverse=True)
    return cands[0][1] if cands else None
def acc_hybrid(seq,vocab,omega,idx,Wq,Wk,Wv,Wo,excluded=set()):
    ok=tot=0
    for step in range(W,len(seq)):
        if seq[step] in seq[max(0,step-W):step]: continue
        if seq[step] in excluded or seq[step-W:step].count(seq[step])>0: continue
        p=predict(seq,vocab,omega,idx,Wq,Wk,Wv,Wo,step,excluded)
        if p is None: continue
        if p==seq[step]: ok+=1
        tot+=1
    return ok/tot if tot else 0.0
def main():
    print("=== v0.14d BORRAR (memoria sobre sustrato que predice) ===")
    seq,vocab=load_seq()
    omega,idx=train_grafo(seq,vocab)
    Wq,Wk,Wv,Wo,_=train_hybrid(seq,vocab,omega,idx)
    base=acc_hybrid(seq,vocab,omega,idx,Wq,Wk,Wv,Wo)
    # nodos top por co-ocurrencia (conceptos compuestos / frecuentes)
    from collections import Counter
    cnt=Counter(seq)
    top=[w for w,_ in cnt.most_common(30)]
    # B: preservar (hibernar) -> omega vive, no se excluye
    acc_b=acc_hybrid(seq,vocab,omega,idx,Wq,Wk,Wv,Wo)
    # D: borrar -> omega=0 y se excluye del vocab del transformer
    omega_d=[list(o) for o in omega]
    excl=set(top)
    for w in top:
        if w in idx: omega_d[idx[w]]=[0.0]*D
    acc_d=acc_hybrid(seq,vocab,omega_d,idx,Wq,Wk,Wv,Wo,excluded=excl)
    out=dict(experiment="v0.14d_borrar_memoria_real",
             hypothesis="Sobre hibrido (predice ~9.6%): borrar nodos top degrada la prediccion; preservarlos no. Memoria real en sustrato con capacidad.",
             params=dict(d=D,V=V,window=W,lr=LR,epochs=EPOCHS,beta=BETA,corpus_n=CORPUS_N,n_borrados=len(top)),
             acc_base=round(base,4), acc_preservado=round(acc_b,4), acc_borrado=round(acc_d,4),
             borrar_destruye=(acc_d < acc_b-0.005))
    json.dump(out,open("results_v14d_borrar.json","w"),indent=2)
    print(f"base={base:.4f} preservado={acc_b:.4f} borrado={acc_d:.4f}")
    print("\n-> results_v14d_borrar.json")
if __name__=="__main__": main()
