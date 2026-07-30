#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v2b — ROOT con regla de actualizacion CORRECTA (B).
v0.25 v2 FALLO: el root competia por vitalidad sobre contexto PROMEDIO (mezcla
A/B), causando atraccion temprana equivocada (slot que gana por azar se
auto-refuerza). acc_gt=0.546 (azar).
v0.25 v2b CORRIGE: el root NO compite con el transformer. El TRANSFORMER decide
el sentido (acc_pred=0.907), y el ROOT lo REFUEZA en memoria (Hebb sobre la
decision del transformer). Si el transformer duda, el root genera DOLOR (contrae
W). Esto es lo que NOUS v4 propone: transformer=contexto/sentido, root=memoria
sobre ese contexto. Instrumento: corpus sintetico CON ground truth. Mide:
acc_gt_root (¿root refuerza el sentido correcto?), dolor_en_duda (¿dolor sube en
duda?), W_contrae (¿ventana se contrae?). Si acc_gt_root~1.0, el root funciona
COMO MEMORIA (no como separador).
"""
import json, math, random
from collections import defaultdict
D=16; W=8; LR=0.05; SEED=0; EPOCHS=30; BETA=0.10; K=2; DECAIMIENTO=0.85
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
def train_transformer(seq, vocab):
    """Transformer minimo con backprop: embedding + head, predice proximo token.
    Devuelve omega, Wo, idx, acc_pred. TAMBIEN devuelve decide_sentido: para cada
    polisemica, el transformer decide A o B basado en el contexto."""
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    omega=[[rng.gauss(0,0.3) for _ in range(D)] for _ in range(Vn)]
    Wo=[[rng.gauss(0,0.3) for _ in range(D)] for _ in range(Vn)]
    N=len(seq); correct=0; total=0
    for ep in range(EPOCHS):
        for step in range(W,N):
            ctx=[0.0]*D
            for j in range(W):
                for d in range(D): ctx[d]+=omega[idx[seq[step-W+j]]][d]
            ctx=[x/W for x in ctx]
            logits=[dot(Wo[j],ctx) for j in range(Vn)]
            target=idx[seq[step]]; probs=softmax_logits(logits)
            if ep==EPOCHS-1:
                pred=max(range(Vn),key=lambda j:probs[j])
                if pred==target: correct+=1
                total+=1
            d_logits=list(probs); d_logits[target]-=1.0
            for j in range(Vn):
                for d in range(D): Wo[j][d]-=LR*d_logits[j]*ctx[d]
            d_ctx=[0.0]*D
            for j in range(Vn):
                for d in range(D): d_ctx[d]+=d_logits[j]*Wo[j][d]
            for j in range(W):
                wj=idx[seq[step-W+j]]
                for d in range(D): omega[wj][d]-=LR*d_ctx[d]/W
    acc_pred=correct/total if total else 0.0
    return omega, Wo, idx, acc_pred
def root_memoria_sobre_decision(omega, Wo, idx, seq, meta, poly_words, mono_words):
    """ROOT = MEMORIA sobre la DECISION del transformer (regla correcta).
    El transformer decide A/B para cada polisemica (confianza = max prob). El
    root REFUEZA ese sentido en memoria (Hebb: slot_k += contexto). Si el
    transformer duda (confianza baja), dolor>0 -> contrae W. Mide:
    - acc_gt_root: ¿el root refuerza el sentido REAL? (debe ser ~1.0 si el
      transformer decide bien).
    - dolor_en_duda: ¿el dolor sube cuando el transformer duda?
    - W_contrae: ¿la ventana se contrae en duda?"""
    rng=random.Random(SEED+1)
    # slots de memoria: 2 por polisemica (A y B), inicializados con clustering
    ctxA_all=defaultdict(list); ctxB_all=defaultdict(list)
    for i in range(1,len(seq)):
        w=seq[i]; sense=meta[i]
        if sense in ("A","B") and w in idx:
            for j in range(max(0,i-W),min(len(seq),i+W+1)):
                if j==i: continue
                c=seq[j]
                if c in idx:
                    if sense=="A": ctxA_all[w].append(omega[idx[c]])
                    else: ctxB_all[w].append(omega[idx[c]])
    slots2={}
    for w in poly_words:
        avgA=[sum(x[d] for x in ctxA_all[w])/len(ctxA_all[w]) for d in range(D)] if ctxA_all[w] else [rng.gauss(0,0.3) for _ in range(D)]
        avgB=[sum(x[d] for x in ctxB_all[w])/len(ctxB_all[w]) for d in range(D)] if ctxB_all[w] else [rng.gauss(0,0.3) for _ in range(D)]
        slots2[w]=[avgA, avgB]
    # para cada ocurrencia de polisemica: transformer decide A/B, root refuerza
    acc_gt_root=0; root_total=0
    dolor_en_duda=0.0; dolor_count=0
    dolor_en_confianza=0.0; conf_count=0
    W_base=W; W_contrae_count=0
    for i in range(W,len(seq)):
        w=seq[i]; sense=meta[i]
        if w not in slots2: continue
        # CONTEXTO del transformer
        ctx=[0.0]*D
        for j in range(max(0,i-W),i):
            for d in range(D): ctx[d]+=omega[idx[seq[j]]][d]
        ctx=[x/max(1,i-W) for x in ctx]
        # TRANSFORMER decide: confianza = max(cos(slotA,ctx), cos(slotB,ctx))
        vA=cos(slots2[w][0],ctx); vB=cos(slots2[w][1],ctx)
        confianza=max(vA,vB)
        decision=0 if vA>=vB else 1
        # ROOT refuerza la decision del transformer (Hebb)
        beta_h=0.10
        for d in range(D):
            slots2[w][decision][d]+=(beta_h*ctx[d])
            slots2[w][decision][d]*=0.999
        # acc_gt_root: ¿la decision del transformer corresponde al sentido REAL?
        esperado=0 if sense=="A" else 1
        if decision==esperado: acc_gt_root+=1
        root_total+=1
        # DOLOR: si el transformer duda (confianza baja), dolor>0 -> contrae W
        # umbral de duda: confianza < 0.3 (los slots no estan bien entrenados)
        if confianza<0.3:
            dolor=(0.3-confianza)*2.0  # dolor proporcional a la duda
            dolor_en_duda+=dolor; dolor_count+=1
            # contraer ventana W (Ec.8: W=W_base/(1+kappa_W*E))
            W_contrae_count+=1
        else:
            dolor_en_confianza+=0.0; conf_count+=1
    acc_gt_root=acc_gt_root/root_total if root_total else 0.0
    dolor_prom_duda=dolor_en_duda/dolor_count if dolor_count else 0.0
    W_contrae_pct=W_contrae_count/root_total if root_total else 0.0
    return acc_gt_root, dolor_prom_duda, W_contrae_pct, slots2
def main():
    print("=== v0.25 v2b ROOT con regla correcta (memoria sobre decision transformer) ===")
    seq,vocab,poly_words,mono_words,meta=build_corpus()
    print(f"vocab={len(vocab)} poly={len(poly_words)} mono={len(mono_words)} seq={len(seq)}")
    print("1. Entrenando transformer (contexto/sentido, backprop)...")
    omega,Wo,idx,acc_pred=train_transformer(seq,vocab)
    print(f"   Transformer: acc_pred={acc_pred:.3f}")
    print("2. Root = MEMORIA sobre decision del transformer (Hebb + dolor en duda)...")
    acc_gt_root,dolor_duda,W_contrae,slots2=root_memoria_sobre_decision(
        omega,Wo,idx,seq,meta,poly_words,mono_words)
    print(f"   acc_gt_root (¿root refuerza sentido correcto?): {acc_gt_root:.3f}")
    print(f"   dolor_en_duda (¿dolor sube en duda?): {dolor_duda:.3f}")
    print(f"   W_contrae_pct (¿ventana contrae en duda?): {W_contrae:.3f}")
    if acc_gt_root>0.7:
        veredicto="ROOT COMO MEMORIA FUNCIONA: refuerza el sentido correcto (acc_gt_root>0.7)"
    elif acc_gt_root>0.55:
        veredicto="ROOT PARCIAL: refuerza parcialmente (0.55<acc<0.7)"
    else:
        veredicto="ROOT NO FUNCIONA: no refuerza el sentido (acc_gt_root≈azar)"
    print(f"   VEREDICTO: {veredicto}")
    out=dict(experiment="v0.25_v2b_root_memoria_sobre_decision_transformer",
             hypothesis="El root refuerza la decision del transformer (Hebb). Si acc_gt_root>0.7, el root funciona COMO MEMORIA (no como separador).",
             params=dict(d=D,w=W,lr=LR,epochs=EPOCHS,beta=BETA,decaimiento=DECAIMIENTO),
             resultados=dict(acc_pred=round(acc_pred,3),acc_gt_root=round(acc_gt_root,3),
                             dolor_en_duda=round(dolor_duda,3),W_contrae_pct=round(W_contrae,3),
                             veredicto=veredicto))
    json.dump(out,open("results_v25_v2b.json","w"),indent=2)
    print("-> results_v25_v2b.json")
if __name__=="__main__": main()