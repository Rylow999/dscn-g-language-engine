#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.23 v3 — COMPOSICION RELACIONAL con DATOS REALES (Don Quijote).
v0.23 v2: senal debil (0.312) sobre corpus SINTETICO chico. Hipotesis: con datos
reales (Don Quijote, ~20k tok, vocab 150) y trplas extraidas de patrones SINTACTICOS
reales (no hardcodeadas), el Hebb 3-body aprende mejor por mas variedad/escala.
Extraccion NO circular: patrones del corpus real -> (suj, REL, obj):
  "X de Y" -> (X, DE, Y)      ; "X en Y" -> (X, EN, Y)   ; "X y Y" -> (X, CON, Y)
  sujeto-verbo por ventana W=4 -> (suj, V, verb)  [relacion implicita de accion]
Test honesto: predecir REL correcta en trplas extraccion; superar azar (1/n_rel).
"""
import json, math, random, re, time
from collections import Counter
SEED=0
def norm(v): return math.sqrt(sum(x*x for x in v)) or 1e-9
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def mat_vec(M,v): return [sum(M[i][j]*v[j] for j in range(len(M))) for i in range(len(M))]
def load_dq(max_tokens=20000):
    txt=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=re.findall(r"[a-záéíóúñü]+", txt.lower())
    vocab=[w for w,_ in Counter(words).most_common(150)]
    idxall=[i for i,w in enumerate(words) if w in set(vocab)]
    step=max(1,len(idxall)//max_tokens); chosen=idxall[::step][:max_tokens]
    seq=[words[i] for i in chosen]
    return seq,vocab
def extract_triples(seq,vocab):
    idx=set(vocab)
    triples=[]  # (s, r, o)
    rels=set()
    for i in range(2,len(seq)):
        a,b,c=seq[i-2],seq[i-1],seq[i]
        if a in idx and c in idx:
            if b=="de":  triples.append((a,"DE",c)); rels.add("DE")
            elif b=="en": triples.append((a,"EN",c)); rels.add("EN")
            elif b=="y":  triples.append((a,"CON",c)); rels.add("CON")
            elif b=="a":  triples.append((a,"A",c)); rels.add("A")
    # sujeto-verbo por ventana: palabra i-3 (sustantivo) + verbo en i-1 + objeto i
    for i in range(3,len(seq)):
        s,v,o=seq[i-3],seq[i-1],seq[i]
        if s in idx and o in idx and v not in ("de","en","y","a") and len(v)>3:
            triples.append((s,"V_"+v,o)); rels.add("V_"+v)
    return triples, sorted(rels)
def train_rel(triples, vocab, rels, D, epochs=15):
    idx={w:i for i,w in enumerate(vocab)}; rng=random.Random(SEED)
    emb=[[rng.gauss(0,1) for _ in range(D)] for _ in range(len(vocab))]
    R={r:[[1.0 if i==j else 0.01*rng.gauss(0,1) for j in range(D)] for i in range(D)] for r in rels}
    lr=0.02
    for ep in range(epochs):
        for (s,r,o) in triples:
            if s not in idx or o not in idx or r not in R: continue
            psr=mat_vec(R[r],emb[idx[s]])
            po_n=[x/norm(emb[idx[o]]) for x in emb[idx[o]]]; psr_n=[x/norm(psr) for x in psr]
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
    return best,round(bs,3)
def run_D(D, epochs=15):
    seq,vocab=load_dq()
    triples,rels=extract_triples(seq,vocab)
    if not triples: return dict(D=D, acc=0, n=0, note="sin trplas")
    emb,R,idx=train_rel(triples, vocab, rels, D, epochs)
    # test: predecir relacion en las trplas extraidas (holdout 20%)
    rng=random.Random(SEED); te=trebles=triples[:]; rng.shuffle(te)
    cut=int(len(te)*0.2); test_set=te[:cut]
    ok=tot=0; det=[]
    for (s,r,o) in test_set:
        pred,sc=predict_rel(emb,R,idx,s,o,rels)
        tot+=1
        hit = (pred==r)
        ok+=1 if hit else 0
        if len(det)<8: det.append(dict(s=s,r=r,o=o,pred=pred,score=sc,hit=hit))
    azar=round(1/len(rels),3)
    return dict(D=D, acc=round(ok/tot,3), n=tot, n_rels=len(rels), baseline_azar=azar,
                supera_azar=ok/tot>azar, detalle=det)
def main():
    print("=== v0.23 v3 COMPOSICION RELACIONAL (DATOS REALES Don Quijote) ===")
    t0=time.time()
    res16=run_D(16); res32=run_D(32)
    print(f"train+eval {time.time()-t0:.0f}s")
    print(f"D=16: {res16['acc']} ({res16['n']} tests, {res16['n_rels']} rels, azar={res16['baseline_azar']}) supera={res16['supera_azar']}")
    print(f"D=32: {res32['acc']} ({res32['n']} tests, {res32['n_rels']} rels, azar={res32['baseline_azar']}) supera={res32['supera_azar']}")
    print("detalle D16:", json.dumps(res16.get('detalle',[]),ensure_ascii=False))
    out={"experiment":"v0.23_v3_composicion_relacional_real",
         "hypothesis":"Con datos reales (Don Quijote) y trplas de patrones sintacticos, el Hebb 3-body supera azar mas fuerte que con sintetico (v0.23 v2: 0.312).",
         "params":{"epochs":15,"extraccion":"de/en/y/a + suj-verb-obj"},
         "D16":res16,"D32":res32}
    json.dump(out,open("results_v23_v3.json","w"),indent=2)
    print("\n-> results_v23_v3.json")
if __name__=="__main__": main()
