#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.12 — ATENCION REAL con ambiguedad (CORREGIDO: palabra DESPUES de banco).
Corpus: [fondo,banca]->"banco"->"dinero"  ;  [madera,silla]->"banco"->"sentar".
"banco" es ambiguo: solo (W=1) no dice si es banca o silla. El contexto (W=2:
contexto+banco) desambigua para predecir la palabra POSTERIOR. Mide si W=2 acierta
mas que W=1 al predecir la palabra tras "banco".
"""
import json, math, random, sys, time
D=8; BETA=0.20; SEED=0
BANCA=["fondo","dinero","cuenta","sucursal","ahorro"]
SILLA=["madera","sentar","comoda","respaldo","rueda"]
def make_corpus(n=6000, rng=None):
    rng=rng or random.Random(SEED)
    seq=[]
    for _ in range(n):
        if rng.random()<0.5:
            c=rng.choice(BANCA); nxt=rng.choice(BANCA)
            seq += [c,"banco",nxt]
        else:
            c=rng.choice(SILLA); nxt=rng.choice(SILLA)
            seq += [c,"banco",nxt]
    vocab=sorted(set(seq))
    return seq, vocab
def omega_of(w,rng):
    r=random.Random(hash(w)%100000); return [r.gauss(0,1) for _ in range(D)]
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    if na<1e-9 or nb<1e-9: return 0.0
    return sum(x*y for x,y in zip(a,b))/(na*nb)
def main():
    print("=== v0.12 ATENCION REAL (ambiguedad, palabra post-banco) ===")
    rng=random.Random(SEED)
    seq,vocab=make_corpus(6000,rng)
    V=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    for i in range(1,len(seq)):
        a,b=seq[i-1],seq[i]; ia,ib=idx[a],idx[b]
        omega[ia]=[(1-BETA)*omega[ia][k]+BETA*omega[ib][k] for k in range(D)]
    def predict(ctx_words, W):
        ctx_omega=[0.0]*D
        for p in ctx_words[-W:]:
            o=omega[idx[p]]
            for k in range(D): ctx_omega[k]+=o[k]
        n=max(1,len(ctx_words[-W:])); ctx=[ctx_omega[k]/n for k in range(D)]
        cands=sorted(((cos(omega[j],ctx),vocab[j]) for j in range(V) if vocab[j] not in ctx_words[-W:]),reverse=True)
        return cands[0][1]
    # test: tras "banco", predecir la palabra correcta (banca o silla segun contexto)
    tests=[]
    for i in range(2,len(seq)):
        if seq[i-1]=="banco":
            tests.append((seq[i-2],"banco",seq[i]))  # contexto, banco, esperado
    ok1=sum(1 for c,b,exp in tests if predict([c,b],1)==exp)  # W=1: solo banco
    ok2=sum(1 for c,b,exp in tests if predict([c,b],2)==exp)  # W=2: contexto+banco
    acc1=ok1/len(tests); acc2=ok2/len(tests)
    out=dict(experiment="v0.12_atencion_real_ambiguedad",
             hypothesis="Con ambiguedad (banco=banca/silla), contexto W=2 acierta mas que bigrama W=1 al predecir palabra post-banco.",
             params=dict(d=D,beta=BETA,vocab=V,corpus="banco ambiguo con post-palabra"),
             n_tests=len(tests), acc_bigrama_W1=round(acc1,4), acc_atencion_W2=round(acc2,4),
             nota="CORREGIDO: mide desambiguar banco usando contexto para predecir lo posterior.")
    with open("results_v12.json","w") as f: json.dump(out,f,indent=2)
    print(f"tests={len(tests)} acc W1={acc1:.4f} acc W2={acc2:.4f}")
    print("\n-> results_v12.json")
if __name__=="__main__": main()
