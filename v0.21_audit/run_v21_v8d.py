#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.21 v8d — COMPETENCIA POR CONTEXTO (regla A).
El fix de v0.21 v8 (anchor+repulsion) FALLO (v8c: acc_gt=0.50, colapso en ctx
fuerte). La causa: la regla de update PROMEDIA vecinos (difusion/oversmoothing),
los 2 subnodos se beben el mismo puré y colapsan.
v0.21 v8d cambia la regla: SOLO se actualiza el subnodo k* que mejor matchea el
contexto actual (competencia). El subnodo no ganador se queda quieto. Esto evita
que los 2 subnodos se mezclen. Instrumento correcto: corpus sintetico CON
ground truth (sentido A/B por ocurrencia) + monosemicas de control + curva
episodio a epoca. Evita circularidad: acc_gt mide contra ground truth REAL, no
contra 'se separo'. Hipotesis: si con competencia acc_gt>0.9 y poli_sep=6/6, el
grafo rustico separa sentidos por contexto SIN transformer.
"""
import json, math, random
from collections import Counter
D=16; W=8; EPOCHS=25; K=2; BETA=0.30; BETA_REP=0.0; ALPHA=0.0; SEED=0
# BETA_REP=0 y ALPHA=0: sin repulsion ni anchor (la competencia sola debe separar)
def norm(v): return math.sqrt(sum(x*x for x in v))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def build_corpus(seed=SEED, n_per_sense=80):
    rng=random.Random(seed)
    poly={
      "banco":  (["dinero","pagar","cuenta","oro","plata","billete","banquero"],
                 ["rio","agua","pez","orilla","puente","corriente","rioja"]),
      "llave":  (["puerta","cerradura","abrir","candado","cerradura","llaves"],
                 ["musica","nota","tono","cancion","melodia","acorde","partitura"]),
      "mouse":  (["computadora","click","pantalla","cable","teclado","raton"],
                 ["animal","cola","hueco","roedor","rato","conejo","campo"]),
      "capital":(["dinero","inversion","bolsa","accion","finanza","mercado"],
                 ["ciudad","pais","provincia","madrid","lima","tierra"]),
      "firma":  (["empresa","contrato","sociedad","negocio","compania"],
                 ["manuscrito","firma","rubrica","documento","dedo","tinta"]),
      "oro":    (["metal","plata","joya","precioso","lingote","moneda"],
                 ["sol","amarillo","brillo","rayo","luz","dorado"]),
    }
    mono={
      "quijote":  ["libro","historia","caballero","mancha","ingenioso","novela"],
      "sancho":   ["escudero","panza","rocinante","insular","gordo","sencillez"],
      "caballero":["andante","espada","honor","armas","valor","caballeria"],
    }
    seq=[]; meta=[]
    def add_block(tokens, sense):
        for t in tokens: seq.append(t); meta.append(sense)
    for w,(sa,sb) in poly.items():
        # INTERCALAR A/B: alternar sentido A y B para que la competencia vea
        # variabilidad real de contexto en la misma palabra (como en corpus real).
        for _ in range(n_per_sense):
            blk=list(sa[:5])+[w]+list(sa[5:7]); add_block(blk,"A")
            blk=list(sb[:5])+[w]+list(sb[5:7]); add_block(blk,"B")
    for w,cm in mono.items():
        for _ in range(n_per_sense):
            blk=list(cm[:5])+[w]+list(cm[5:7]); add_block(blk,"M")
    return seq, list(dict.fromkeys(seq)), list(poly.keys()), list(mono.keys()), meta
def eval_full(seq, vocab, idx, frac, meta, poly_words, mono_words):
    """acc_gt: ¿subnodo ganador corresponde al sentido real? ps/ms: separaciones."""
    correctos=0; total=0
    poli_sep=0; mono_sep=0
    # poli_sep: ¿cada polisémica se separa (<85% en un bucket)?
    for w in poly_words:
        if w not in idx: continue
        wi=idx[w]; occ=[i for i,x in enumerate(seq) if x==w]
        if len(occ)<10: continue
        grupos={}
        for i in occ:
            cw=list(range(max(0,i-W),i))
            if not cw: continue
            ctx=[0.0]*D
            for c in cw:
                o=frac[idx[seq[c]]][0]
                for d in range(D): ctx[d]+=o[d]
            ctx=[x/len(cw) for x in ctx]
            bestk,bestc=-1,-1e9
            for k in range(K):
                c=cos(frac[wi][k],ctx)
                if c>bestc: bestc=c; bestk=k
            grupos.setdefault(bestk,0); grupos[bestk]+=1
        n=len(occ); dom=max(grupos.values()) if grupos else n
        if len(grupos)>=2 and dom<n*0.85: poli_sep+=1
    for w in mono_words:
        if w not in idx: continue
        wi=idx[w]; occ=[i for i,x in enumerate(seq) if x==w]
        if len(occ)<10: continue
        grupos={}
        for i in occ:
            cw=list(range(max(0,i-W),i))
            if not cw: continue
            ctx=[0.0]*D
            for c in cw:
                o=frac[idx[seq[c]]][0]
                for d in range(D): ctx[d]+=o[d]
            ctx=[x/len(cw) for x in ctx]
            bestk,bestc=-1,-1e9
            for k in range(K):
                c=cos(frac[wi][k],ctx)
                if c>bestc: bestc=c; bestk=k
            grupos.setdefault(bestk,0); grupos[bestk]+=1
        n=len(occ); dom=max(grupos.values()) if grupos else n
        if len(grupos)>=2 and dom<n*0.85: mono_sep+=1
    # acc_gt: para polisemicas, ¿el subnodo ganador corresponde al sentido real?
    for i in range(1,len(seq)):
        w=seq[i]; sense=meta[i]
        if sense not in ("A","B") or w not in idx: continue
        wi=idx[w]; cw=list(range(max(0,i-W),i))
        if not cw: continue
        ctx=[0.0]*D
        for c in cw:
            o=frac[idx[seq[c]]][0]
            for d in range(D): ctx[d]+=o[d]
        ctx=[x/len(cw) for x in ctx]
        bestk,bestc=-1,-1e9
        for k in range(K):
            c=cos(frac[wi][k],ctx)
            if c>bestc: bestc=c; bestk=k
        esperado=0 if sense=="A" else 1
        if bestk==esperado: correctos+=1
        total+=1
    return (correctos/total if total else 0.0), poli_sep, mono_sep
def train_compet(seq, vocab, poly_words, mono_words, meta):
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    frac=[[[rng.gauss(0,1) for _ in range(D)] for _ in range(K)] for _ in range(Vn)]
    curve=[]
    for ep in range(EPOCHS):
        for i in range(1,len(seq)):
            a=idx[seq[i-1]]; b=idx[seq[i]]
            cw=list(range(max(0,i-1-W),i-1))
            if not cw: continue
            ctx=[0.0]*D
            for c in cw:
                o=frac[idx[seq[c]]][0]
                for d in range(D): ctx[d]+=o[d]
            ctx=[x/len(cw) for x in ctx]
            bestk,bestc=-1,-1e9
            for k in range(K):
                c=cos(frac[a][k],ctx)
                if c>bestc: bestc=c; bestk=k
            k=bestk
            new=[(1-BETA)*frac[a][k][d]+BETA*frac[b][0][d] for d in range(D)]
            frac[a][k]=new
        acc,ps,ms=eval_full(seq,vocab,idx,frac,meta,poly_words,mono_words)
        curve.append((ep,round(acc,3),ps,ms))
        if ep%5==0: print(f"  ep{ep:2} acc_gt={acc:.3f} poli_sep={ps}/{len(poly_words)} mono_sep={ms}/{len(mono_words)}")
    return frac, idx, curve
def main():
    print("=== v0.21 v8d COMPETENCIA POR CONTEXTO (regla A) ===")
    seq,vocab,poly_words,mono_words,meta=build_corpus()
    print(f"vocab={len(vocab)} poly={len(poly_words)} mono={len(mono_words)} seq={len(seq)}")
    frac,idx,curve=train_compet(seq,vocab,poly_words,mono_words,meta)
    acc,ps,ms=eval_full(seq,vocab,idx,frac,meta,poly_words,mono_words)
    print(f"FINAL: acc_gt={acc:.3f} poli_sep={ps}/{len(poly_words)} mono_sep={ms}/{len(mono_words)}")
    print(f"curva: {curve}")
    out=dict(experiment="v0.21_v8d_competencia_por_contexto",
             hypothesis="Competencia (solo subnodo k* ganador se actualiza) separa polisemicas SIN mezclar subnodos. acc_gt>0.9 => grafo rustico viable.",
             params=dict(d=D,w=W,beta=BETA,beta_rep=BETA_REP,alpha=ALPHA,epochs=EPOCHS),
             resultados=dict(acc_gt=round(acc,3),poli_sep=ps,mono_sep=ms,
                             curve=curve))
    json.dump(out,open("results_v21_v8d.json","w"),indent=2)
    print("-> results_v21_v8d.json")
if __name__=="__main__": main()