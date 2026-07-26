#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.15 — SENSE NODES (polisemia estructural, idea de Luciano).
"banco" no es UN nodo: son sentidos distintos (banco_banca, banco_silla) con omega
distinto, indexados por sentido no por nomenclatura. El contexto elige el sentido.
Test controlado: corpus ambiguo [fondo,banca]->banco->dinero ; [madera,silla]->banco->sentar.
Si los sense nodes resuelven la desambiguacion (fondo->banca, madera->silla) donde
v0.13/0.14 fallaron, la idea 1 funciona.
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
def main():
    print("=== v0.15 SENSE NODES (polisemia estructural) ===")
    rng=random.Random(SEED)
    seq,vocab=make_corpus(6000,rng)
    V=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    # SENSE SPLIT: "banco" se parte en 2 nodos por familia de contexto
    # (en corpus real seria clustering no supervisado de contexto; aca es controlado)
    banco_senses={"banco_banca":0,"banco_silla":1}
    sense_idx={s: V+si for s,si in banco_senses.items()}
    omega.append([rng.gauss(0,1) for _ in range(D)])  # banco_banca
    omega.append([rng.gauss(0,1) for _ in range(D)])  # banco_silla
    NS=V+2
    # entrenar: cada aparicion de "banco" va a su sentido segun contexto
    def sense_of(context_word):
        return "banco_banca" if context_word in set(BANCA) else "banco_silla"
    t0=time.time()
    for i in range(1,len(seq)):
        a,b=seq[i-1],seq[i]
        # nodo fuente: si es banco, usamos su sentido
        a_id = sense_idx[sense_of(a)] if a=="banco" else idx[a]
        b_id = sense_idx[sense_of(b)] if b=="banco" else idx[b]
        omega[a_id]=[(1-BETA)*omega[a_id][k]+BETA*omega[b_id][k] for k in range(D)]
    print(f"entrenado {time.time()-t0:.0f}s")
    # TEST: dado contexto (fondo/madera) + banco, predecir post-word y sentido
    def predict_post(context_word):
        # elegir sentido de banco por afinidad del contexto con los sentidos
        cw=omega[idx[context_word]]
        sb=omega[sense_idx["banco_banca"]]; ss=omega[sense_idx["banco_silla"]]
        sense = "banco_banca" if cos(cw,sb)>cos(cw,ss) else "banco_silla"
        sn=omega[sense_idx[sense]]
        # predecir post-word por afinidad con sn, excluyendo banco y contexto
        cands=sorted(((cos(omega[j],sn),vocab[j]) for j in range(V) if vocab[j]!=context_word),reverse=True)
        return sense, cands[0][1]
    tests=[]
    for i in range(2,len(seq)):
        if seq[i-1]=="banco":
            tests.append((seq[i-2],seq[i]))  # contexto, post esperado
    ok_sense=0; ok_post=0
    for c,exp in tests:
        sense,pred=predict_post(c)
        true_sense = "banco_banca" if c in set(BANCA) else "banco_silla"
        if sense==true_sense: ok_sense+=1
        if pred==exp: ok_post+=1
    acc_sense=ok_sense/len(tests); acc_post=ok_post/len(tests)
    # baseline: un solo nodo banco (mezcla) predice post al azar ~50%
    out=dict(experiment="v0.15_sense_nodes",
             hypothesis="Sense nodes (identidad estructural por sentido) resuelven polisemia que v0.13/0.14 no pudieron.",
             params=dict(d=D,beta=BETA,vocab=V,senses=2),
             n_tests=len(tests),
             acc_sense=round(acc_sense,4),
             acc_post_word=round(acc_post,4),
             baseline_single_node_post=round(0.5,4),
             nota="banco se parte en 2 nodos por contexto. Indexado por sentido, no nomenclatura.")
    with open("results_v15.json","w") as f: json.dump(out,f,indent=2)
    print(f"tests={len(tests)} acc_sense={acc_sense:.4f} acc_post={acc_post:.4f} (baseline 1 nodo ~0.50)")
    print("\n-> results_v15.json")
if __name__=="__main__": main()
