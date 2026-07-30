#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v2d — ¿LA REPRESENTACION DEL TRANSFORMER SEPARA A/B?
B: el root no separa sentido (acc_gt≈azar en v0.22 v2, v0.25 v2, v2b, v2c).
¿es imposible o mal camino? v2d pregunta: ¿la representacion omega del transformer
separan A/B para cada polisemica? Si separan, el root puede aprender a leerlos
(no imposible). Si no separan, el transformer no separa sentido en su
representacion (acc_pred=0.907 es solo prediccion de tokens, no sentido).
Reutiliza train_transformer + build_corpus de v0.25 v2c.
"""
import json, math, random
from collections import defaultdict
exec(open("run_v25_v2c.py").read().split("def main")[0])  # importa D/W/LR/etc + funciones
def separa_AB(omega, idx, seq, meta, poly_words):
    """¿la representacion omega del transformer separa A/B para cada polisemia?
    Para cada polisemica: construye avgA, avgB (promedio de embeddings de tokens
    de contexto A/B). Mide cos(avgA, avgB): si es bajo (distintos), el transformer
    separa sentido en su representacion. Tambien acc_gt_simple: ¿el contexto
    actual se clasifica correctamente A/B usando omega?"""
    sa_tokens=defaultdict(set); sb_tokens=defaultdict(set)
    for i in range(1,len(seq)):
        w=seq[i]; sense=meta[i]
        if sense in ("A","B") and w in idx:
            for j in range(max(0,i-W),min(len(seq),i+W+1)):
                if j==i: continue
                c=seq[j]
                if c in idx:
                    if sense=="A": sa_tokens[w].add(c)
                    else: sb_tokens[w].add(c)
    resultados={}
    acc_total=0; acc_n=0
    for w in poly_words:
        if w not in idx: continue
        # avgA, avgB = promedio de embeddings de tokens de contexto A/B
        ctxsA=[omega[idx[c]] for c in sa_tokens[w] if c in idx]
        ctxsB=[omega[idx[c]] for c in sb_tokens[w] if c in idx]
        if not ctxsA or not ctxsB: continue
        avgA=[sum(x[d] for x in ctxsA)/len(ctxsA) for d in range(D)]
        avgB=[sum(x[d] for x in ctxsB)/len(ctxsB) for d in range(D)]
        sep=cos(avgA,avgB)  # bajo = separan bien
        # acc_gt_simple: para cada ocurrencia, ¿el contexto actual es mas similar a avgA o avgB?
        occ=[i for i,x in enumerate(seq) if x==w]
        correctos=0; total=0
        for i in occ:
            if meta[i] not in ("A","B"): continue
            cw=list(range(max(0,i-W),i))
            if not cw: continue
            ctx=[0.0]*D
            for c in cw:
                if c in idx:
                    for d in range(D): ctx[d]+=omega[idx[c]][d]
            ctx=[x/max(1,len(cw)) for x in ctx]
            vA=cos(avgA,ctx); vB=cos(avgB,ctx)
            bestk=0 if vA>=vB else 1
            esperado=0 if meta[i]=="A" else 1
            if bestk==esperado: correctos+=1
            total+=1
        acc=correctos/total if total else 0.0
        resultados[w]=dict(cos_AB=round(sep,3),acc_gt_simple=round(acc,3))
        acc_total+=acc; acc_n+=1
    acc_prom=acc_total/acc_n if acc_n else 0.0
    return resultados, acc_prom
def main():
    print("=== v0.25 v2d ¿omega del transformer separa A/B? ===")
    seq,vocab,poly_words,mono_words,meta=build_corpus()
    print(f"vocab={len(vocab)} poly={len(poly_words)} mono={len(mono_words)} seq={len(seq)}")
    omega,Wo,idx,acc_pred=train_transformer(seq,vocab)
    print(f"Transformer: acc_pred={acc_pred:.3f}")
    res,acc_prom=separa_AB(omega,idx,seq,meta,poly_words)
    print(f"acc_gt_simple (clasificacion A/B con omega): {acc_prom:.3f} (azar=0.50)")
    for w,r in res.items():
        sep=r['cos_AB']
        print(f"  {w}: cos(A,B)={sep:.3f} acc_gt={r['acc_gt_simple']:.3f} "
              f"{'(SEPARA)' if sep<0.5 and r['acc_gt_simple']>0.7 else '(NO SEPARA)'}")
    if acc_prom>0.7:
        veredicto="TRANSFORMER SEPARA A/B: el root puede aprender a leerlos (no imposible)"
    elif acc_prom>0.55:
        veredicto="TRANSFORMER SEPARA PARCIALMENTE A/B"
    else:
        veredicto="TRANSFORMER NO SEPARA A/B (acc≈azar): acc_pred=0.907 es prediccion de tokens, no sentido"
    print(f"VEREDICTO: {veredicto}")
    out=dict(experiment="v0.25_v2d_omega_separa_AB",
             hypothesis="La representacion omega del transformer separa A/B. Si acc_gt_simple>0.7, el root puede aprender a leerlos.",
             params=dict(d=D,w=W,lr=LR,epochs=EPOCHS),
             resultados=dict(acc_pred=round(acc_pred,3),acc_gt_simple=round(acc_prom,3),
                             detalle=res,veredicto=veredicto))
    json.dump(out,open("results_v25_v2d.json","w"),indent=2)
    print("-> results_v25_v2d.json")
if __name__=="__main__": main()