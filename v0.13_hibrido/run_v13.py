#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.13 HIBRIDO grafo + ATENCION APRENDIDA (resuelve desambig. de v0.12).
El grafo tiene omega_base por token (memoria/hibernado). La atencion produce una
representacion CONTEXTUALIZADA: omega_ctx(banco) = omega_banco + sum_i a_i*(omega_ci - omega_banco),
donde a_i = softmax(omega_banco . omega_ci / sqrt(D)). Asi "banco" tras "fondo"
queda en otro punto del espacio que tras "madera" (atencion modula, no solo promedia).
Corpus ambiguo: [fondo,banca]->banco->dinero ; [madera,silla]->banco->sentar.
Mide: ¿la atencion contextualizada acierta mas que bigrama (W=1) y que promedio (v0.12)?
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
            c=rng.choice(BANCA); nxt=rng.choice(BANCA); seq += [c,"banco",nxt]
        else:
            c=rng.choice(SILLA); nxt=rng.choice(SILLA); seq += [c,"banco",nxt]
    return seq, sorted(set(seq))
def omega_of(w,rng):
    r=random.Random(hash(w)%100000); return [r.gauss(0,1) for _ in range(D)]
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    if na<1e-9 or nb<1e-9: return 0.0
    return sum(x*y for x,y in zip(a,b))/(na*nb)

def attention_ctx(omega, idx, ctx_words, target="banco"):
    # representacion contextualizada del target dado el contexto
    wt=omega[idx[target]]
    keys=[omega[idx[w]] for w in ctx_words]
    # softmax de afinidad
    logits=[cos(wt,k)/math.sqrt(D) for k in keys]
    mx=max(logits); ex=[math.exp(l-mx) for l in logits]; s=sum(ex); att=[e/s for e in ex]
    ctx=[0.0]*D
    for a,k in zip(att,keys):
        for d in range(D): ctx[d]+=a*k[d]
    # modulo: omega_ctx = omega_target + sum a_i*(key_i - omega_target)
    out=[wt[d]+sum(att[i]*(keys[i][d]-wt[d]) for i in range(len(keys))) for d in range(D)]
    return out

def main():
    print("=== v0.13 HIBRIDO grafo + atencion aprendida (desambig. banco) ===")
    rng=random.Random(SEED)
    seq,vocab=make_corpus(6000,rng)
    V=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    # entrenar omega_base por next-token (banco aprende acercarse a contexto y post)
    for i in range(1,len(seq)):
        a,b=seq[i-1],seq[i]; ia,ib=idx[a],idx[b]
        omega[ia]=[(1-BETA)*omega[ia][k]+BETA*omega[ib][k] for k in range(D)]
    def predict(ctx_words, target, mode):
        if mode=="W1":
            ctx=omega[idx[target]]  # solo banco
        elif mode=="avg":
            ctx=[0.0]*D
            for w in ctx_words+[target]:
                o=omega[idx[w]]
                for d in range(D): ctx[d]+=o[d]
            n=len(ctx_words+[target]); ctx=[c/n for c in ctx]
        else:  # att: contextualizado
            ctx=attention_ctx(omega, idx, ctx_words, target)
        cands=sorted(((cos(omega[j],ctx),vocab[j]) for j in range(V) if vocab[j] not in ctx_words[-2:]),reverse=True)
        return cands[0][1]
    tests=[]
    for i in range(2,len(seq)):
        if seq[i-1]=="banco":
            tests.append((seq[i-2],"banco",seq[i]))
    ok_w1=sum(1 for c,b,exp in tests if predict([c],b,"W1")==exp)
    ok_avg=sum(1 for c,b,exp in tests if predict([c],b,"avg")==exp)
    ok_att=sum(1 for c,b,exp in tests if predict([c],b,"att")==exp)
    a1=ok_w1/len(tests); a2=ok_avg/len(tests); a3=ok_att/len(tests)
    out=dict(experiment="v0.13_hibrido_atencion",
             hypothesis="Atencion aprendida (representacion contextualizada de banco) desambigua mejor que bigrama/promedio.",
             params=dict(d=D,beta=BETA,vocab=V,corpus="banco ambiguo"),
             n_tests=len(tests), acc_W1=round(a1,4), acc_avg_v012=round(a2,4), acc_atencion=round(a3,4),
             nota="Hibrido: grafo (omega_base) + atencion que MODULA omega por contexto.")
    with open("results_v13.json","w") as f: json.dump(out,f,indent=2)
    print(f"tests={len(tests)} W1={a1:.4f} avg(v0.12)={a2:.4f} ATENCION={a3:.4f}")
    print("\n-> results_v13.json")

if __name__=="__main__": main()
