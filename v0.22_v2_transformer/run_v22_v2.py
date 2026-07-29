#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.22 v2 — ROOT DIRECTOR sobre TRANSFORMER (v0.14d).
v0.21 v8→v8f CERRARON que el grafo rustico D=16 no separa sentidos (acc_gt<=0.53,
azar). La causa: no entrena embeddings de co-ocurrencia real (no hay backprop).
v0.22 v2 re-define sobre el TRANSFORMER v0.14d (que SI separa sentidos vía
backprop). La pregunta REAL de v0.22: ¿la proyeccion Hebb del ROOT DIRECTOR
aporta sobre un sustrato que separa sentidos? (no sobre el grafo rustico).
Diseño: transformer minimo (embedding + atencion + head, backprop) sobre corpus
sintetico CON ground truth (sentido A/B). Luego proyeccion Hebb del root sobre
los embeddings del transformer. Mide: routing_acc_gt = ¿el root rutea al subnodo
correcto SEGUN ground truth? Si >0.7, la proyeccion Hebb aporta. Si ~0.50, el
root solo refleja el transformer (no aporta).
"""
import json, math, random
from collections import Counter, defaultdict
D=16; W=8; LR=0.05; SEED=0; EPOCHS=30; BETA=0.10; K=2
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def mat_vec(M,v): return [dot(M[i],v) for i in range(len(M))]
def vec_add(a,b,a2=1.0): return [a[i]+a2*b[i] for i in range(len(a))]
def scale(v,s): return [x*s for x in v]
def softmax_logits(logits):
    mx=max(logits); ex=[math.exp(l-mx) for l in logits]; s=sum(ex); return [e/s for e in ex]
def norm(v): return math.sqrt(sum(x*x for x in v))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def build_corpus(seed=SEED, n_per_sense=60):
    """Corpus sintetico con ground truth (sentido A/B). Polisemias con contextos
    DISTINTOS para A y B (para que el transformer pueda aprender a distinguirlos)."""
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
    """Transformer MINIMO con backprop COMPLETO (v0.14d-style).
    Entrena embeddings que separan sentidos: embedding (omega) + head (Wo),
    backprop para predecir el proximo token. La atencion simplificada (promedio
    de contexto) basta para separar A/B en corpus sintetico con contextos
    distintos. Devuelve: omega (embeddings entrenados), Wo (head), idx.
    Instrumento: tambien devuelve acc_pred (accuracy de prediccion del proximo
    token) para verificar que el transformer SÍ aprende."""
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    omega=[[rng.gauss(0,0.3) for _ in range(D)] for _ in range(Vn)]
    Wo=[[rng.gauss(0,0.3) for _ in range(D)] for _ in range(Vn)]
    N=len(seq); correct=0; total=0
    for ep in range(EPOCHS):
        for step in range(W,N):
            # contexto: promedio de embeddings de las W palabras anteriores
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
            # backprop Wo
            for j in range(Vn):
                for d in range(D): Wo[j][d]-=LR*d_logits[j]*ctx[d]
            # backprop ctx -> omega (propaga a las W palabras del contexto)
            d_ctx=[0.0]*D
            for j in range(Vn):
                for d in range(D): d_ctx[d]+=d_logits[j]*Wo[j][d]
            for j in range(W):
                wj=idx[seq[step-W+j]]
                for d in range(D): omega[wj][d]-=LR*d_ctx[d]/W
    acc_pred=correct/total if total else 0.0
    return omega, Wo, idx, acc_pred
def root_proyeccion(omega, idx, seq, meta, poly_words):
    """PROYECCION HEBB del ROOT DIRECTOR (SIN backprop, perfil DSCN-G).
    Para cada palabra, construye K=2 subnodos proyectando el embedding sobre
    los contextos A y B (Hebb: subnodo_k = omega[w] proyectado en direccion
    del contexto k). El root elige el subnodo que mejor matchea el contexto
    actual. Mide routing_acc_gt: ¿el subnodo elegido corresponde al sentido
    REAL (ground truth)?"""
    # construir contexto promedio A/B por palabra (como ancla de proyeccion)
    ctxA=defaultdict(list); ctxB=defaultdict(list)
    for i in range(1,len(seq)):
        w=seq[i]; sense=meta[i]
        if sense in ("A","B") and w in idx:
            for j in range(max(0,i-W),min(len(seq),i+W+1)):
                if j==i: continue
                c=seq[j]
                if c in idx:
                    if sense=="A": ctxA[w].append(omega[idx[c]])
                    else: ctxB[w].append(omega[idx[c]])
    avgA={}; avgB={}
    for w in poly_words:
        avgA[w]=[sum(x[d] for x in ctxA[w])/len(ctxA[w]) for d in range(D)] if ctxA[w] else [0.0]*D
        avgB[w]=[sum(x[d] for x in ctxB[w])/len(ctxB[w]) for d in range(D)] if ctxB[w] else [0.0]*D
    # subnodos Hebb: subA = omega[w] proyectado en avgA, subB = omega[w] proyectado en avgB
    # (proyeccion Hebb: componente de omega[w] en la direccion del contexto)
    subnodos={}
    for w in poly_words:
        if w not in idx: continue
        base=omega[idx[w]]
        # proyeccion Hebb: subA = base + beta * avgA (refuerza direccion A)
        subA=vec_add(base, avgA[w], BETA)
        subB=vec_add(base, avgB[w], BETA)
        subnodos[w]=[subA, subB]
    # routing_acc_gt: para cada ocurrencia de w, ¿el subnodo ganador corresponde al sentido?
    correctos=0; total=0
    for i in range(1,len(seq)):
        w=seq[i]; sense=meta[i]
        if sense not in ("A","B") or w not in subnodos: continue
        cw=list(range(max(0,i-W),i))
        if not cw: continue
        ctx=[0.0]*D
        for c in cw:
            if c in idx:
                for d in range(D): ctx[d]+=omega[idx[c]][d]
        ctx=[x/len(cw) for x in ctx]
        bestk,bestc=-1,-1e9
        for k in range(K):
            c=cos(subnodos[w][k],ctx)
            if c>bestc: bestc=c; bestk=k
        esperado=0 if sense=="A" else 1
        if bestk==esperado: correctos+=1
        total+=1
    return correctos/total if total else 0.0, subnodos, avgA, avgB
def baseline_contexto(seq, vocab, idx, meta, poly_words):
    """BASELINE: ¿el contexto SOLO (sin root) rutea al sentido correcto?
    Para cada ocurrencia de w, el subnodo ganador = el contexto promedio A o B
    que mejor matchea el contexto actual. Si esto da >0.7, el transformer solo
    ya separa sentidos y el root no aporta. Si <0.7, el root podría aportar."""
    ctxA=defaultdict(list); ctxB=defaultdict(list)
    for i in range(1,len(seq)):
        w=seq[i]; sense=meta[i]
        if sense in ("A","B") and w in idx:
            for j in range(max(0,i-W),min(len(seq),i+W+1)):
                if j==i: continue
                c=seq[j]
                if c in idx:
                    if sense=="A": ctxA[w].append(idx[c])
                    else: ctxB[w].append(idx[c])
    # baseline: para cada ocurrencia, ¿el contexto actual es más similar a avgA o avgB?
    correctos=0; total=0
    for i in range(1,len(seq)):
        w=seq[i]; sense=meta[i]
        if sense not in ("A","B") or w not in ctxA or w not in ctxB: continue
        if not ctxA[w] or not ctxB[w]: continue
        cw=[idx[c] for c in seq[max(0,i-W):i] if c in idx]
        if not cw: continue
        # contexto actual = conjunto de indices de palabras vecinas
        ctx_set=set(cw)
        simA=len(ctx_set & set(ctxA[w]))/len(ctx_set)
        simB=len(ctx_set & set(ctxB[w]))/len(ctx_set)
        bestk=0 if simA>=simB else 1
        esperado=0 if sense=="A" else 1
        if bestk==esperado: correctos+=1
        total+=1
    return correctos/total if total else 0.0
def main():
    print("=== v0.22 v2 ROOT DIRECTOR sobre TRANSFORMER (v0.14d) ===")
    seq,vocab,poly_words,mono_words,meta=build_corpus()
    print(f"vocab={len(vocab)} poly={len(poly_words)} mono={len(mono_words)} seq={len(seq)}")
    print("Entrenando transformer minimo (backprop completo)...")
    omega,Wo,idx,acc_pred=train_transformer(seq,vocab)
    print(f"Transformer entrenado. acc_pred(next-token)={acc_pred:.3f} (azar=1/{len(vocab)}={1/len(vocab):.3f})")
    # routing_acc_gt del ROOT sobre el transformer
    root_acc,subnodos,avgA,avgB=root_proyeccion(omega,idx,seq,meta,poly_words)
    # baseline: contexto solo (sin root)
    base_acc=baseline_contexto(seq,vocab,idx,meta,poly_words)
    print(f"routing_acc_gt (root sobre transformer): {root_acc:.3f}")
    print(f"baseline (contexto solo, sin root):      {base_acc:.3f}")
    if root_acc>base_acc+0.05:
        veredicto="ROOT APORTE: root > baseline"
    elif root_acc<base_acc-0.05:
        veredicto="ROOT NO APORTE: root < baseline"
    else:
        veredicto="ROOT REFLEJA: root ~= baseline (no aporta)"
    print(f"VEREDICTO: {veredicto}")
    out=dict(experiment="v0.22_v2_root_sobre_transformer",
             hypothesis="La proyeccion Hebb del root aporta sobre embeddings del transformer (root_acc > baseline). Si root~=baseline, el root solo refleja el transformer.",
             params=dict(d=D,w=W,lr=LR,epochs=EPOCHS,beta=BETA,k=K),
             resultados=dict(acc_pred=round(acc_pred,3),root_acc=round(root_acc,3),
                             baseline_acc=round(base_acc,3),veredicto=veredicto))
    json.dump(out,open("results_v22_v2.json","w"),indent=2)
    print("-> results_v22_v2.json")
if __name__=="__main__": main()