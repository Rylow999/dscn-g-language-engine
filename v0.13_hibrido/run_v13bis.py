#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.13-bis HIBRIDO grafo + atencion sobre Don Quijote (vocab 150, corpus REAL).
El corpus sintetico de v0.13 colapsaba los contextos (fondo/madera -> mismo omega).
Aca usamos Don Quijote (vocab 150) donde cada palabra TIENE identidad propia.
Prediccion con atencion contextualizada: estado = softmax(afinidad) sobre omega del
contexto de W palabras. Mide si W=2,3 (atencion) supera W=1 (bigrama, 10.11% de v0.6a).
"""
import json, math, random, re, sys, time
D=8; BETA=0.20; V=150; STEPS=4000; SEED=0

def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words, V):
    from collections import Counter
    return [w for w,_ in Counter(words).most_common(V)]
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    if na<1e-9 or nb<1e-9: return 0.0
    return sum(x*y for x,y in zip(a,b))/(na*nb)

def main():
    print("=== v0.13-bis HIBRIDO atencion sobre Don Quijote (vocab 150) ===")
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text); vocab=build_vocab(words,V)
    rng=random.Random(SEED)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    seq=[w for w in words if w in set(vocab)]
    idx={w:i for i,w in enumerate(vocab)}
    t0=time.time()
    for i in range(1,len(seq)):
        a,b=seq[i-1],seq[i]; ia,ib=idx[a],idx[b]
        omega[ia]=[(1-BETA)*omega[ia][k]+BETA*omega[ib][k] for k in range(D)]
    print(f"entrenado {time.time()-t0:.0f}s")
    def predict(ctx_words, W):
        # atencion: estado = sum a_i * omega[ctx_i], a_i=softmax(cos(omega[ult],omega[ctx_i]))
        if not ctx_words: return None
        ult=omega[idx[ctx_words[-1]]]
        ctx=ctx_words[-W:]
        keys=[omega[idx[w]] for w in ctx]
        logits=[cos(ult,k)/math.sqrt(D) for k in keys]
        mx=max(logits); ex=[math.exp(l-mx) for l in logits]; s=sum(ex); att=[e/s for e in ex]
        state=[0.0]*D
        for a,k in zip(att,keys):
            for d in range(D): state[d]+=a*k[d]
        # excluir contexto inmediato del candidato
        excl=set(ctx_words[-W:])
        cands=sorted(((cos(omega[j],state),vocab[j]) for j in range(V) if vocab[j] not in excl),reverse=True)
        return cands[0][1]
    def acc(W):
        ok=0; tot=0
        for i in range(W,len(seq)):
            ctx=seq[i-W:i]; exp=seq[i]
            if exp in ctx: continue
            if predict(ctx,W)==exp: ok+=1
            tot+=1
        return ok/tot
    a1=acc(1); a2=acc(2); a3=acc(3)
    out=dict(experiment="v0.13bis_hibrido_atencion_donquijote",
             hypothesis="Atencion contextualizada sobre corpus real (Don Quijote) supera bigrama (10.11% v0.6a).",
             params=dict(d=D,beta=BETA,V=V,steps=STEPS,corpus="don_quijote"),
             acc_bigrama_W1=round(a1,4), acc_atencion_W2=round(a2,4), acc_atencion_W3=round(a3,4),
             baseline_v06a=0.1011,
             nota="v0.13 (corpus sintetico) colapsaba contextos; aca usamos corpus real con identidad.")
    with open("results_v13bis.json","w") as f: json.dump(out,f,indent=2)
    print(f"acc W1={a1:.4f}  W2={a2:.4f}  W3={a3:.4f}  (baseline v0.6a=0.1011)")
    print("\n-> results_v13bis.json")

if __name__=="__main__": main()
