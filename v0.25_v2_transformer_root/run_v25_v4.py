#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v4 — ROOT como SISTEMA DE DUDA sobre el transformer.
v0.25 v2d: el transformer mínimo separa tokens (acc_pred=0.907) PERO NO sentidos
(cos(A,B) alto, acc_gt≈azar). El root no separa sentido (5 experimentos, acc_gt≈0.50).
PERO el root funciona COMO SISTEMA DE DUDA (v0.25 v2c: dolor_duda=0.841, W=0.982).
v0.25 v4 valida el root como sistema de duda sobre el transformer: para cada
palabra, el transformer decide sentido (Wo); el root mide COHERENCIA entre la
decision y la memoria (slots con vitalidad). Si coherencia baja (duda), dolor>0
-> contrae W (Ec.8). Mide: dolor_en_ambiguedad (¿subte en contextos realmente
ambiguos?), W_contrae (¿contrae la ventana?), foco_acc (¿retiene lo relevante?).
La pregunta de tu mama ("Messi se activa sin contexto") implica que el root no
crea sentido (el transformer lo hace) sino que opera como sistema de duda.
"""
import json, math, random
from collections import defaultdict
D=16; W=8; LR=0.05; SEED=0; EPOCHS=30; BETA=0.10; DECAIMIENTO=0.85; KAPPA_W=2.0
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
    """Transformer minimo (next-token, backprop). Mismo que v0.25 v2d."""
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
            # LayerNorm
            for j in range(W):
                wj=idx[seq[step-W+j]]
                n=norm(omega[wj])
                if n>1e-9:
                    for d in range(D): omega[wj][d]/=n
    acc_pred=correct/total if total else 0.0
    return omega, Wo, idx, acc_pred
def root_sistema_duda(omega, Wo, idx, seq, meta, poly_words, mono_words, vocab):
    """ROOT = SISTEMA DE DUDA sobre el transformer.
    Para cada palabra polisemica: el transformer decide sentido (Wo predice el
    proximo token; confianza = max prob). El root mide COHERENCIA entre la
    decision y la memoria (slots con vitalidad). Si coherencia baja (duda),
    dolor>0 -> contrae W (Ec.8). Mide:
    - dolor_en_ambiguedad: ¿el dolor sube en contextos realmente ambiguos?
    - W_contrae_pct: ¿la ventana se contrae en duda?
    - foco_acc: ¿la vitalidad retiene lo relevante?
    - acc_decision_transformer: ¿el transformer decide bien el sentido?"""
    rng=random.Random(SEED+1)
    # slots de memoria: 2 por polisemica (A y B), inicializados con clustering
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
    slots2={}
    for w in poly_words:
        ctxsA=[omega[idx[c]] for c in sa_tokens[w] if c in idx]
        ctxsB=[omega[idx[c]] for c in sb_tokens[w] if c in idx]
        avgA=[sum(x[d] for x in ctxsA)/len(ctxsA) for d in range(D)] if ctxsA else [rng.gauss(0,0.3) for _ in range(D)]
        avgB=[sum(x[d] for x in ctxsB)/len(ctxsB) for d in range(D)] if ctxsB else [rng.gauss(0,0.3) for _ in range(D)]
        slots2[w]=[avgA, avgB]
    # para cada ocurrencia de polisemica: transformer decide + root mide duda
    dolor_en_ambiguedad=0.0; duda_count=0
    dolor_en_confianza=0.0; conf_count=0
    W_contrae_count=0; root_total=0
    acc_decision=0; decision_total=0
    foco_correct=0; foco_total=0
    for i in range(W,len(seq)):
        w=seq[i]; sense=meta[i]
        if w not in slots2: continue
        # CONTEXTO del transformer
        ctx=[0.0]*D
        for j in range(max(0,i-W),i):
            for d in range(D): ctx[d]+=omega[idx[seq[j]]][d]
        ctx=[x/max(1,i-W) for x in ctx]
        # TRANSFORMER decide: Wo predice proximo token, confianza = max prob
        logits=[dot(Wo[j],ctx) for j in range(len(vocab))]
        probs=softmax_logits(logits)
        confianza=max(probs)
        pred_idx=max(range(len(vocab)),key=lambda j:probs[j])
        pred_token=vocab[pred_idx]
        # decision del transformer: A si predice token de sentido A, B si de B
        if pred_token in sa_tokens[w]:
            decision_t=0
        elif pred_token in sb_tokens[w]:
            decision_t=1
        else:
            # token no distintivo: usar cos con slots
            vA=cos(slots2[w][0],ctx); vB=cos(slots2[w][1],ctx)
            decision_t=0 if vA>=vB else 1
        esperado=0 if sense=="A" else 1
        if decision_t==esperado: acc_decision+=1
        decision_total+=1
        # ROOT mide COHERENCIA: ¿la decision del transformer coincide con el
        # slot ganador por vitalidad? Si no coinciden, hay dolor (duda)
        vA=cos(slots2[w][0],ctx); vB=cos(slots2[w][1],ctx)
        slot_ganador=0 if vA>=vB else 1
        coherencia=1.0 if slot_ganador==decision_t else 0.0
        # DOLOR: incoherencia entre decision del transformer y memoria del root
        dolor=1.0-coherencia
        # contraer ventana W (Ec.8: W=W_base/(1+kappa_W*dolor))
        W_actual=W/(1.0+KAPPA_W*dolor)
        if W_actual<W: W_contrae_count+=1
        root_total+=1
        # clasificar dolor: en duda (confianza baja) vs confianza alta
        if confianza<0.1:  # duda del transformer
            dolor_en_ambiguedad+=dolor; duda_count+=1
        else:
            dolor_en_confianza+=dolor; conf_count+=1
        # foco: ¿el slot ganador del root corresponde al sentido real?
        if slot_ganador==esperado: foco_correct+=1
        foco_total+=1
    acc_decision=acc_decision/decision_total if decision_total else 0.0
    dolor_duda=dolor_en_ambiguedad/duda_count if duda_count else 0.0
    dolor_conf=dolor_en_confianza/conf_count if conf_count else 0.0
    W_contrae_pct=W_contrae_count/root_total if root_total else 0.0
    foco_acc=foco_correct/foco_total if foco_total else 0.0
    return dict(acc_decision=acc_decision, dolor_en_duda=dolor_duda,
                dolor_en_confianza=dolor_conf, W_contrae_pct=W_contrae_pct,
                foco_acc=foco_acc, duda_count=duda_count, conf_count=conf_count)
def main():
    print("=== v0.25 v4 ROOT como SISTEMA DE DUDA sobre transformer ===")
    seq,vocab,poly_words,mono_words,meta=build_corpus()
    print(f"vocab={len(vocab)} poly={len(poly_words)} mono={len(mono_words)} seq={len(seq)}")
    print("1. Entrenando transformer (contexto/sentido)...")
    omega,Wo,idx,acc_pred=train_transformer(seq,vocab)
    print(f"   Transformer: acc_pred={acc_pred:.3f}")
    print("2. Root = SISTEMA DE DUDA (coherencia decision vs memoria)...")
    r=root_sistema_duda(omega,Wo,idx,seq,meta,poly_words,mono_words,vocab)
    print(f"   acc_decision (transformer decide sentido?): {r['acc_decision']:.3f}")
    print(f"   dolor_en_duda (¿subte en contextos ambiguos?): {r['dolor_en_duda']:.3f} ({r['duda_count']} casos)")
    print(f"   dolor_en_confianza (¿bajo en contextos claros?): {r['dolor_en_confianza']:.3f} ({r['conf_count']} casos)")
    print(f"   W_contrae_pct (¿ventana contrae en duda?): {r['W_contrae_pct']:.3f}")
    print(f"   foco_acc (¿memoria retiene lo relevante?): {r['foco_acc']:.3f}")
    if r['dolor_en_duda']>r['dolor_en_confianza'] and r['W_contrae_pct']>0.5:
        veredicto="ROOT COMO DUDA FUNCIONA: dolor sube en ambiguidad, contrae W"
    else:
        veredicto="ROOT COMO DUDA NO FUNCIONA: dolor no distingue ambiguidad"
    print(f"   VEREDICTO: {veredicto}")
    out=dict(experiment="v0.25_v4_root_sistema_de_duda",
             hypothesis="El root mide coherencia entre decision del transformer y memoria. Si dolor_en_duda>dolor_en_confianza y W_contrae, el root funciona como sistema de duda.",
             params=dict(d=D,w=W,lr=LR,epochs=EPOCHS,beta=BETA,decaimiento=DECAIMIENTO,kappa_w=KAPPA_W),
             resultados=r)
    out["resultados"]["veredicto"]=veredicto
    json.dump(out,open("results_v25_v4.json","w"),indent=2)
    print("-> results_v25_v4.json")
if __name__=="__main__": main()