#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.15-bis — SENSE NODES con contextos de IDENTIDAD REAL (corrige v0.15).
El fallo de v0.15 fue el corpus: "fondo"/"madera" colapsaban a omega igual porque
solo precedian a "banco". Aca los contextos APARECEN EN VARIAS POSICIONES (tienen
identidad propia), asi el sentido de "banco" se elige bien por afinidad real.
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
            # contexto aparece tambien en otra posicion (identidad)
            seq += [rng.choice(BANCA),"otra",c]
        else:
            c=rng.choice(SILLA); nxt=rng.choice(SILLA)
            seq += [c,"banco",nxt]
            seq += [rng.choice(SILLA),"otra",c]
    return seq, sorted(set(seq))
def omega_of(w,rng):
    r=random.Random(hash(w)%100000); return [r.gauss(0,1) for _ in range(D)]
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    if na<1e-9 or nb<1e-9: return 0.0
    return sum(x*y for x,y in zip(a,b))/(na*nb)
def main():
    print("=== v0.15-bis SENSE NODES (contextos con identidad) ===")
    rng=random.Random(SEED)
    seq,vocab=make_corpus(6000,rng)
    V=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    sense_idx={"banco_banca":V, "banco_silla":V+1}
    omega.append([rng.gauss(0,1) for _ in range(D)])
    omega.append([rng.gauss(0,1) for _ in range(D)])
    NS=V+2
    def sense_of(w): return "banco_banca" if w in set(BANCA) else "banco_silla"
    t0=time.time()
    for i in range(1,len(seq)):
        a,b=seq[i-1],seq[i]
        a_id=sense_idx[sense_of(a)] if a=="banco" else idx[a]
        b_id=sense_idx[sense_of(b)] if b=="banco" else idx[b]
        omega[a_id]=[(1-BETA)*omega[a_id][k]+BETA*omega[b_id][k] for k in range(D)]
    print(f"entrenado {time.time()-t0:.0f}s")
    def predict(context_word):
        cw=omega[idx[context_word]]
        sb=omega[sense_idx["banco_banca"]]; ss=omega[sense_idx["banco_silla"]]
        sense="banco_banca" if cos(cw,sb)>cos(cw,ss) else "banco_silla"
        sn=omega[sense_idx[sense]]
        cands=sorted(((cos(omega[j],sn),vocab[j]) for j in range(V) if vocab[j]!=context_word),reverse=True)
        return sense, cands[0][1]
    tests=[]
    for i in range(2,len(seq)):
        if seq[i-1]=="banco":
            tests.append((seq[i-2],seq[i]))
    ok_s=ok_p=0
    for c,exp in tests:
        sense,pred=predict(c)
        ts="banco_banca" if c in set(BANCA) else "banco_silla"
        if sense==ts: ok_s+=1
        if pred==exp: ok_p+=1
    acc_s=ok_s/len(tests); acc_p=ok_p/len(tests)
    out=dict(experiment="v0.15bis_sense_nodes_identidad",
             hypothesis="Con contextos de identidad real, sense nodes eligen el sentido correcto (resuelve polisemia).",
             params=dict(d=D,beta=BETA,vocab=V,senses=2,corpus="contextos en varias posiciones"),
             n_tests=len(tests), acc_sense=round(acc_s,4), acc_post=round(acc_p,4),
             baseline_v15=0.4987, nota="v0.15 fallo por corpus colapsado; aca contextos tienen identidad.")
    with open("results_v15bis.json","w") as f: json.dump(out,f,indent=2)
    print(f"tests={len(tests)} acc_sense={acc_s:.4f} acc_post={acc_p:.4f} (v0.15 daba 0.4987)")
    print("\n-> results_v15bis.json")
if __name__=="__main__": main()
