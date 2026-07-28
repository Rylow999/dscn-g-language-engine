#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.23 v2 — COMPOSICION RELACIONAL sin contaminacion + D=16/32 + mas steps.
v0.23 v1: 0.333 (< azar 0.5). Causa: al acercar emb[s]~emb[o] (asociacion basica)
se contamina R[TIENE] y R[LUGAR] (ambos pares ocurren -> colapsan). FIX: NO acercar
emb[s],emb[o]; solo reforzar R[r] tal que R[r]*emb[s] ~ emb[o]. Ademas: corpus menos
sintetico (mas sujetos/objetos/relaciones variadas) y mas steps de Hebb.
Test honesto: predecir relacion correcta entre sujeto y objeto, superar azar (0.5).
"""
import json, math, random, time
from collections import Counter
SEED=0
def norm(v): return math.sqrt(sum(x*x for x in v)) or 1e-9
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def mat_vec(M,v): return [sum(M[i][j]*v[j] for j in range(len(M))) for i in range(len(M))]
def build_corpus(seed=SEED, n=20):
    rng=random.Random(seed)
    # relaciones variadas: TIENE, LUGAR, CAUSA, PARTE_DE
    # cada sujeto tiene objetos tipicos de CADA relacion (poco solapados)
    data={
      "banco":  {"TIENE":["dinero","cuenta","oro"], "LUGAR":["ciudad","plaza","calle"]},
      "casa":   {"TIENE":["puerta","techo","llave"], "LUGAR":["barrio","vereda","esquina"]},
      "arbol":  {"TIENE":["hoja","fruto","rama"], "LUGAR":["bosque","suelo","huerta"]},
      "fuego":  {"TIENE":["llama","ceniza","calor"], "CAUSA":["leña","chispa","viento"]},
      "lluvia": {"TIENE":["gota","charco","agua"], "CAUSA":["nube","tormenta","humedad"]},
      "reloj":  {"TIENE":["manecilla","engranaje","esfera"], "PARTE_DE":["muñeca","pared","mesa"]},
      "libro":  {"TIENE":["pagina","tapa","indice"], "PARTE_DE":["estante","mochila","mano"]},
      "coche":  {"TIENE":["rueda","motor","puerta"], "PARTE_DE":["garaje","calle","ruta"]},
    }
    rels=["TIENE","LUGAR","CAUSA","PARTE_DE"]
    seq=[]; rel=[]
    for s,d in data.items():
        for r in rels:
            if r not in d: continue
            for _ in range(n):
                o=rng.choice(d[r]); seq+=["el",s,r.lower(),o]; rel+=["X",r,r,r]
    pr=list(zip(seq,rel)); rng.shuffle(pr); seq=[p[0] for p in pr]; rel=[p[1] for p in pr]
    vocab=list(dict.fromkeys(seq))
    return seq,rel,vocab,data,rels
def train_rel(seq,rel,vocab,rels,D,epochs=20):
    global mat_vec
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}; rng=random.Random(SEED)
    emb=[[rng.gauss(0,1) for _ in range(D)] for _ in range(Vn)]
    R={r:[[1.0 if i==j else 0.01*rng.gauss(0,1) for j in range(D)] for i in range(D)] for r in rels}
    lr=0.02
    for ep in range(epochs):
        for i in range(2,len(seq)):
            if rel[i] in R and seq[i-2] in idx and seq[i] in idx:
                s=idx[seq[i-2]]; o=idx[seq[i]]; r=rel[i]
                psr=mat_vec(R[r],emb[s])
                po_n=[x/norm(emb[o]) for x in emb[o]]; psr_n=[x/norm(psr) for x in psr]
                # SOLO refuerza R[r]: R[r] += lr * outer(psr_n, po_n)  (SIN acercar emb[s],emb[o])
                R[r]=[[R[r][a][b]+lr*psr_n[a]*po_n[b] for b in range(D)] for a in range(D)]
    return emb,R,idx
def predict_rel(emb,R,idx,s,o,rels):
    best=None; bs=-2.0
    for r in rels:
        if s not in idx or o not in idx: return None,-2.0
        psr=mat_vec(R[r],emb[idx[s]])
        sc=cos(psr,emb[idx[o]])
        if sc is None: sc=-2.0
        if sc>bs: bs=sc; best=r
    return best, round(bs,3)
def run_D(D, epochs=20):
    global mat_vec
    seq,rel,vocab,data,rels=build_corpus(n=20)
    emb,R,idx=train_rel(seq,rel,vocab,rels,D,epochs)
    test={}
    for s,d in data.items():
        for r in rels:
            if r not in d: continue
            for o in d[r][:2]:
                pred,sc=predict_rel(emb,R,idx,s,o,rels)
                test[f"{s}-{o}"]=dict(pred=pred,score=sc,gt=r)
    ok=sum(1 for v in test.values() if v["pred"]==v["gt"]); tot=len(test)
    # baseline azar: 4 relaciones -> 0.25
    return dict(D=D, acc=round(ok/tot,3), n=tot, baseline_azar=round(1/len(rels),3),
                supera_azar=ok/tot>1/len(rels), detalle=test)
def main():
    print("=== v0.23 v2 COMPOSICION RELACIONAL (sin contaminacion, D=16/32) ===")
    t0=time.time()
    res16=run_D(16, epochs=20)
    res32=run_D(32, epochs=20)
    print(f"train+eval {time.time()-t0:.0f}s")
    print(f"D=16: {res16['acc']} ({res16['n']} tests, azar={res16['baseline_azar']}) supera_azar={res16['supera_azar']}")
    print(f"D=32: {res32['acc']} ({res32['n']} tests, azar={res32['baseline_azar']}) supera_azar={res32['supera_azar']}")
    out={"experiment":"v0.23_v2_composicion_relacional",
         "hypothesis":"Sin asociacion basica contaminante + corpus variado + mas steps, R[r] aprende relaciones (4 tipos). Debe superar azar (0.25 con 4 relaciones).",
         "params":{"epochs":20,"rels":["TIENE","LUGAR","CAUSA","PARTE_DE"]},
         "D16":res16,"D32":res32}
    json.dump(out,open("results_v23_v2.json","w"),indent=2)
    print("\n-> results_v23_v2.json")
if __name__=="__main__": main()
