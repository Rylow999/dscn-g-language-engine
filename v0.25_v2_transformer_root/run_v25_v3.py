#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v3 — TRANSFORMER BERT-STYLE (masked LM).
v0.25 v2d: transformer next-token separa tokens (acc_pred=0.907) PERO NO sentidos
(cos(A,B)=0.57-0.79, acc_gt=0.533≈azar). La pregunta de tu mamá: el cerebro
separa "Messi" SIN contexto → necesitamos embeddings entrenados con supervicion
de sentido (no solo prediccion de tokens).
v0.25 v3: transformer con MASKED LM (borrar token, predecir). El masked LM
fuerza a aprender representaciones sensibles al contexto → deberia separar A/B.
Mide: acc_pred(MLM), cos(A,B) (¿separan?), acc_gt_simple (¿clasifica A/B?).
Si separa, el root puede operar sobre esta representacion.
"""
import json, math, random
from collections import defaultdict
D=16; W=8; LR=0.05; SEED=0; EPOCHS=40; BETA=0.10; MASK_RATE=0.15
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def softmax_logits(logits):
    mx=max(logits); ex=[math.exp(l-mx) for l in logits]; s=sum(ex); return [e/s for e in ex]
def norm(v): return math.sqrt(sum(x*x for x in v))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def build_corpus(seed=SEED, n_per_sense=60):
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
      "oro":    (["metal","plata","joya","precioso","lingote","moneda"],
                 ["sol","amarillo","brillo","rayo","luz","dorado"]),
    }
    mono={
      "quijote":  ["libro","historia","caballero","mancha","ingenioso","novela"],
      "sancho":   ["escudero","panza","rocinante","insular","gordo","sencillez"],
    }
    seq=[]; meta=[]
    def add_block(tokens, sense):
        for t in tokens: seq.append(t); meta.append(sense)
    for w,(sa,sb) in poly.items():
        for _ in range(n_per_sense):
            blk=list(sa[:5])+[w]+list(sa[5:7]); add_block(blk,"A")
            blk=list(sb[:5])+[w]+list(sb[5:7]); add_block(blk,"B")
    for w,cm in mono.items():
        for _ in range(n_per_sense):
            blk=list(cm[:5])+[w]+list(cm[5:7]); add_block(blk,"M")
    return seq, list(dict.fromkeys(seq)), list(poly.keys()), list(mono.keys()), meta
def train_bert(seq, vocab):
    """Transformer BERT-STYLE: embedding + atencion (Wq/Wk/Wv) + head Wo.
    Entrenamiento: MASKED LM (borrar MASK_RATE de tokens, predecir con Wo).
    El masked LM fuerza a aprender representaciones sensibles al contexto →
    deberia separar A/B. Devuelve omega (embeddings), Wo (head), idx, acc_mlm."""
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    omega=[[rng.gauss(0,0.3) for _ in range(D)] for _ in range(Vn)]
    Wo=[[rng.gauss(0,0.3) for _ in range(D)] for _ in range(Vn)]
    # atencion: Wq/Wk/Wv (D x D)
    def rnd_mat(): return [[rng.gauss(0,0.3) for _ in range(D)] for _ in range(D)]
    Wq=rnd_mat(); Wk=rnd_mat(); Wv=rnd_mat()
    N=len(seq); correct=0; total=0
    for ep in range(EPOCHS):
        for step in range(W,N):
            # MASKED LM: borrar MASK_RATE de posiciones, predecir con contexto
            mask_pos=set()
            for j in range(W):
                if rng.random()<MASK_RATE:
                    mask_pos.add(j)
            if not mask_pos: continue
            # contexto: embeddings de las W palabras (masked -> embedding neutro)
            ctx=[0.0]*D
            for j in range(W):
                if j in mask_pos:
                    v=[0.0]*D  # mascara: embedding neutro
                else:
                    v=omega[idx[seq[step-W+j]]]
                for d in range(D): ctx[d]+=v[d]
            ctx=[x/W for x in ctx]
            # atencion: Q=ctx, K/V = embeddings de contexto
            q=ctx  # query = contexto promedio
            h=[0.0]*D
            for j in range(W):
                k_vec=omega[idx[seq[step-W+j]]]
                v_vec=omega[idx[seq[step-W+j]]]
                score=dot(q,k_vec)/math.sqrt(D)
                att=score/math.sqrt(D)  # atencion simple (no softmax, para rapidez)
                for d in range(D): h[d]+=att*v_vec[d]
            h=[x/W for x in h]
            logits=[dot(Wo[j],h) for j in range(Vn)]
            probs=softmax_logits(logits)
            # para cada posicion enmascarada, predecir el token real
            for j in mask_pos:
                target=idx[seq[step-W+j]]
                pred=max(range(Vn),key=lambda jj:probs[jj])
                if ep==EPOCHS-1:
                    if pred==target: correct+=1
                    total+=1
                d_logits=list(probs); d_logits[target]-=1.0
                # backprop Wo
                for jj in range(Vn):
                    for d in range(D): Wo[jj][d]-=LR*d_logits[jj]*h[d]
                # backprop h -> omega (atribuye a los tokens del contexto)
                d_h=[0.0]*D
                for jj in range(Vn):
                    for d in range(D): d_h[d]+=d_logits[jj]*Wo[jj][d]
                for j2 in range(W):
                    if j2 in mask_pos: continue
                    wj=idx[seq[step-W+j2]]
                    for d in range(D): omega[wj][d]-=LR*d_h[d]/W
            # LayerNorm: normalizar embeddings para evitar colapso
            for j2 in range(W):
                wj=idx[seq[step-W+j2]]
                n=norm(omega[wj])
                if n>1e-9:
                    for d in range(D): omega[wj][d]/=n
    acc_mlm=correct/total if total else 0.0
    return omega, Wo, idx, acc_mlm
def separa_AB(omega, idx, seq, meta, poly_words):
    """¿la representacion omega del BERT separa A/B? Mide cos(A,B) y acc_gt_simple."""
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
        ctxsA=[omega[idx[c]] for c in sa_tokens[w] if c in idx]
        ctxsB=[omega[idx[c]] for c in sb_tokens[w] if c in idx]
        if not ctxsA or not ctxsB: continue
        avgA=[sum(x[d] for x in ctxsA)/len(ctxsA) for d in range(D)]
        avgB=[sum(x[d] for x in ctxsB)/len(ctxsB) for d in range(D)]
        sep=cos(avgA,avgB)
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
    return resultados, acc_total/acc_n if acc_n else 0.0
def main():
    print("=== v0.25 v3 TRANSFORMER BERT-STYLE (masked LM) ===")
    seq,vocab,poly_words,mono_words,meta=build_corpus()
    print(f"vocab={len(vocab)} poly={len(poly_words)} mono={len(mono_words)} seq={len(seq)}")
    print("Entrenando BERT-style (masked LM)...")
    omega,Wo,idx,acc_mlm=train_bert(seq,vocab)
    print(f"Transformer BERT: acc_mlm={acc_mlm:.3f} (azar=1/{len(vocab)}={1/len(vocab):.3f})")
    res,acc_prom=separa_AB(omega,idx,seq,meta,poly_words)
    print(f"acc_gt_simple (¿clasifica A/B?): {acc_prom:.3f} (azar=0.50)")
    for w,r in res.items():
        print(f"  {w}: cos(A,B)={r['cos_AB']:.3f} acc_gt={r['acc_gt_simple']:.3f} "
              f"{'(SEPARA)' if r['cos_AB']<0.5 and r['acc_gt_simple']>0.7 else '(NO SEPARA)'}")
    if acc_prom>0.7:
        veredicto="BERT SEPARA A/B: el root puede operar sobre esta representacion"
    elif acc_prom>0.55:
        veredicto="BERT SEPARA PARCIALMENTE A/B"
    else:
        veredicto="BERT NO SEPARA A/B: masked LM no basta, necesitamos mas supervicion"
    print(f"VEREDICTO: {veredicto}")
    out=dict(experiment="v0.25_v3_bert_style_masked_lm",
             hypothesis="El transformer BERT-style (masked LM) separa A/B. Si acc_gt_simple>0.7, el root puede operar sobre esta representacion.",
             params=dict(d=D,w=W,lr=LR,epochs=EPOCHS,mask_rate=MASK_RATE),
             resultados=dict(acc_mlm=round(acc_mlm,3),acc_gt_simple=round(acc_prom,3),
                             detalle=res,veredicto=veredicto))
    json.dump(out,open("results_v25_v3.json","w"),indent=2)
    print("-> results_v25_v3.json")
if __name__=="__main__": main()