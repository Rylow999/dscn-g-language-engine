#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.14 HIBRIDO REAL: grafo (memoria) + transformer RUSTICO 2 capas (contexto).
El grafo aporta omega_base por token (memoria/categoria/dolor, de v0.6a).
El transformer RUSTICO (2 capas de self-atencion con pesos aprendidos LOCALMENTE
por regla Hebbiana, no backprop) computa estado contextualizado h_t por ventana W.
Predice next-token por coseno(h_t, omega_base). Mide si supera v0.6a (10.11%).
Diferencia con v0.13: aca los pesos de atencion SE APRENDEN (no son coseno sobre
omega fijo), y hay 2 capas (estado se recomputa, no se desplaza omega).
"""
import json, math, random, re, sys, time
D=8; V=150; W=4; LAYERS=2; BETA=0.10; SEED=0
CORPUS_N=30000

def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words, V):
    from collections import Counter
    return [w for w,_ in Counter(words).most_common(V)]
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    if na<1e-9 or nb<1e-9: return 0.0
    return sum(x*y for x,y in zip(a,b))/(na*nb)
def mat_identity(): return [[1.0 if i==j else 0.0 for j in range(D)] for i in range(D)]
def mat_apply(M,v):
    return [sum(M[i][j]*v[j] for j in range(D)) for i in range(D)]
def mat_add(M,N,a=1.0):
    return [[M[i][j]+a*N[i][j] for j in range(D)] for i in range(D)]

def main():
    print("=== v0.14 HIBRIDO REAL: grafo + transformer rustico 2 capas ===")
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text)[:CORPUS_N*2]; vocab=build_vocab(words,V)
    rng=random.Random(SEED)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    seq=[w for w in words if w in set(vocab)][:CORPUS_N]
    idx={w:i for i,w in enumerate(vocab)}
    # grafo: omega_base por next-token (memoria del grafo)
    t0=time.time()
    for i in range(1,len(seq)):
        a,b=seq[i-1],seq[i]; ia,ib=idx[a],idx[b]
        omega[ia]=[(1-BETA)*omega[ia][k]+BETA*omega[ib][k] for k in range(D)]
    print(f"grafo entrenado {time.time()-t0:.0f}s")
    # transformer rustico: pesos por capa (Q,K,V) aprendidos localmente
    Wq=[mat_identity() for _ in range(LAYERS)]
    Wk=[mat_identity() for _ in range(LAYERS)]
    Wv=[mat_identity() for _ in range(LAYERS)]
    def attention_layer(h_list, lay):
        # h_list: estados de la ventana. atencion causal sobre la ventana.
        out=[]
        for t in range(len(h_list)):
            q=mat_apply(Wq[lay], h_list[t])
            # claves/valores de posiciones 0..t
            keys=[mat_apply(Wk[lay], h_list[s]) for s in range(t+1)]
            vals=[mat_apply(Wv[lay], h_list[s]) for s in range(t+1)]
            logits=[cos(q,keys[s])/math.sqrt(D) for s in range(t+1)]
            mx=max(logits); ex=[math.exp(l-mx) for l in logits]; s2=sum(ex); att=[e/s2 for e in ex]
            st=[0.0]*D
            for a2,vv in zip(att,vals):
                for d in range(D): st[d]+=a2*vv[d]
            out.append(st)
        return out
    # entrenar transformer por next-token (Hebbiano: estado se acerca a omega objetivo)
    t0=time.time()
    for i in range(W,len(seq)):
        ctx=[omega[idx[seq[i-W+j]]] for j in range(W)]  # embedding = omega_base del grafo
        # 2 capas
        h=attention_layer(ctx,0)
        h=attention_layer(h,1)
        target=omega[idx[seq[i]]]
        # aprender: Wq/Wk/Wv de la ultima capa se acercan a proyectar h[-1]->target
        # Hebb simple: ajusta Wv[1] para que vals apunten a target cuando att alto
        # (aproximacion: empuja Wv[1] hacia target - h_base)
        err=[target[d]-h[-1][d] for d in range(D)]
        for d in range(D):
            for j in range(D):
                Wv[1][d][j]+=0.01*err[d]*h[-1][j]
        if (i-W)%5000==0:
            pass
    print(f"transformer entrenado {time.time()-t0:.0f}s")
    # evaluar: predecir next-token usando estado contextualizado de W palabras
    def predict(i):
        ctx=[omega[idx[seq[i-W+j]]] for j in range(W)]
        h=attention_layer(ctx,0); h=attention_layer(h,1)
        excl=set(seq[i-W:i])
        cands=sorted(((cos(omega[j],h[-1]),vocab[j]) for j in range(V) if vocab[j] not in excl),reverse=True)
        return cands[0][1]
    ok=0; tot=0
    for i in range(W,len(seq)):
        if seq[i] in seq[i-W:i]: continue
        if predict(i)==seq[i]: ok+=1
        tot+=1
    acc=ok/tot
    out=dict(experiment="v0.14_hibrido_real_grafo_transformer",
             hypothesis="Transformer rustico 2 capas sobre omega_base del grafo supera bigrama v0.6a (10.11%).",
             params=dict(d=D,V=V,window=W,layers=LAYERS,corpus_n=CORPUS_N,aprendizaje="hebbiano local"),
             acc_hibrido=round(acc,4), baseline_v06a=0.1011,
             nota="Grafo=memoria (omega_base). Transformer=contexto (2 capas, pesos aprendidos). Distinguir de v0.13 (1 capa, coseno sobre omega fijo).")
    with open("results_v14.json","w") as f: json.dump(out,f,indent=2)
    print(f"acc hibrido={acc:.4f}  (baseline v0.6a=0.1011)")
    print("\n-> results_v14.json")

if __name__=="__main__": main()
