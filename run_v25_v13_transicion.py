#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v13 — MODELO DE TRANSICION EXPLICITO (bigramas/trigramas).
Objetivo: capturar estructura de transicion del lenguaje sin depender de
embeddings. Compara contra v12 (decoder por similitud de embeddings).

Metodo:
  - Entrenar bigramas: P(w_next | w_prev)
  - Entrenar trigramas: P(w_next | w_prev2, w_prev1)
  - Generar desde muestra o beam search.
  - Medir top-1/top-5 precision.
"""
import json, math, random, re
from collections import defaultdict, Counter

random.seed(0)

POLYSEMY = {
    "banco": {
        "A": ["dinero","pagar","cuenta","ahorro","plata","banquero","interes","cheque","tarjeta","retiro"],
        "B": ["rio","agua","pez","orilla","puente","corriente","boga","remo","proa","popa"],
        "templates_A": [
            "fue al banco para dinero y pagar con tarjeta en mano",
            "el banco aprobo el interes sin plazo ni comision",
            "si tienes ahorro en el banco podras usar el cheque sin credito",
            "cerro su cuenta en el banco despues de retirar el saldo",
            "el banco publico ajusto la tasa de interes por la inflacion",
            "acredite el dinero en el banco para evitar el robo",
        ],
        "templates_B": [
            "se tiro al banco del rio para pescar con su red",
            "el bote choco contra el banco de la orilla al remar",
            "amarraron la barca en el banco mientras el agua subia",
            "cerca del banco se pesco una trucha sobre la arena",
            "el puente esta sobre el banco para cruzarlo temprano",
            "bajamos por el banco del rio hasta la playa",
        ],
    },
}

def fill_template(tpl, sense, word):
    def rep(m):
        key=m.group(0).strip("{}")
        idx=int(key[1:])-1
        return POLYSEMY[word][sense][idx % len(POLYSEMY[word][sense])]
    return re.sub(r'\{[AB]\d+\}', rep, tpl)

def build_corpus(n_per_sense=350, word="banco"):
    seq=[]; meta=[]
    rng=random.Random(0)
    for sense_label in ["A","B"]:
        tpls=POLYSEMY[word][f"templates_{sense_label}"]
        for _ in range(n_per_sense):
            sentence=random.choice(tpls)
            toks=sentence.split()
            seq.extend(toks); meta.extend([sense_label if word in toks else "O" for _ in toks])
    return seq, meta, word

class TransicionModel:
    def __init__(self, order=2):
        self.order=order
        self.bi=defaultdict(Counter)      # P(w|w_prev)
        self.tri=defaultdict(lambda: defaultdict(int))  # P(w|w_prev2,w_prev1)
    def fit(self, seq):
        for i in range(len(seq)-1):
            w=seq[i]; wn=seq[i+1]
            self.bi[w][wn]+=1
            if self.order>=2 and i>=1:
                w2=seq[i-1]
                self.tri[(w2,w)][wn]+=1
        # normalizar
        self.bi_probs={}
        for w,c in self.bi.items():
            total=sum(c.values())
            self.bi_probs[w]={k:v/total for k,v in c.items()}
        self.tri_probs={}
        for key,c in self.tri.items():
            total=sum(c.values())
            self.tri_probs[key]={k:v/total for k,v in c.items()}
    def predict(self, ctx, top_k=5):
        if len(ctx)>=2 and tuple(ctx[-2:]) in self.tri_probs:
            probs=self.tri_probs[tuple(ctx[-2:])]
        elif ctx[-1] in self.bi_probs:
            probs=self.bi_probs[ctx[-1]]
        else:
            return []
        return sorted(probs, key=probs.get, reverse=True)[:top_k]
    def generate(self, context_words, max_len=15, temperature=1.0):
        generated=[]
        ctx=list(context_words)
        for _ in range(max_len):
            preds=self.predict(ctx, top_k=10)
            if not preds: break
            if temperature==0:
                nxt=preds[0]
            else:
                if tuple(ctx[-2:]) in self.tri_probs:
                    probs=self.tri_probs[tuple(ctx[-2:])]
                elif ctx[-1] in self.bi_probs:
                    probs=self.bi_probs[ctx[-1]]
                else:
                    probs={p:1/len(preds) for p in preds}
                scores=[probs.get(w,0) for w in preds]
                max_s=max(scores) if scores else 1.0
                exps=[math.exp((s-max_s)/temperature) for s in scores]
                total=sum(exps)
                if total<1e-9:
                    nxt=preds[0]
                else:
                    probs_norm=[e/total for e in exps]
                    r=random.random(); cum=0; nxt=preds[0]
                    for w,p in zip(preds, probs_norm):
                        cum+=p
                        if r<=cum:
                            nxt=w
                            break
            generated.append(nxt)
            ctx.append(nxt)
            ctx=ctx[-10:]
        return generated

def evaluar(model, seq, W=8, n_samples=200):
    correct1=0; correct5=0; total=0
    rng=random.Random(3)
    indices=rng.sample(range(W, len(seq)), min(n_samples, len(seq)-W))
    for i in indices:
        ctx=seq[max(0,i-W):i]
        preds=model.predict(ctx, top_k=5)
        target=seq[i]
        if preds and preds[0]==target: correct1+=1
        if target in preds: correct5+=1
        total+=1
    return correct1/total if total>0 else 0.0, correct5/total if total>0 else 0.0, total

def main():
    print("=== v0.25 v13 MODELO DE TRANSICION EXPLICITO ===")
    seq,meta,word=build_corpus(n_per_sense=350)
    print(f" seq={len(seq)} vocab={len(set(seq))}")

    # orden 1
    print(" entrenando bigramas...")
    m1=TransicionModel(order=1); m1.fit(seq)
    top1_1, top5_1, n1 = evaluar(m1, seq, W=8, n_samples=200)
    print(f" bigramas: top1={top1_1:.3f} top5={top5_1:.3f} (n={n1})")

    # orden 2
    print(" entrenando trigramas...")
    m2=TransicionModel(order=2); m2.fit(seq)
    top1_2, top5_2, n2 = evaluar(m2, seq, W=8, n_samples=200)
    print(f" trigramas: top1={top1_2:.3f} top5={top5_2:.3f} (n={n2})")

    # generaciones
    prompts=[
        ["fue","al","banco"],
        ["el","banco","aprobo"],
        ["se","tiro","al","banco"],
        ["el","bote","choco"],
        ["la","llave","de","la"],
    ]
    gen1=[(p, m1.generate(p, max_len=10, temperature=0.5)) for p in prompts]
    gen2=[(p, m2.generate(p, max_len=10, temperature=0.5)) for p in prompts]

    print("\n generaciones:")
    for (p,g1),(_,g2) in zip(gen1,gen2):
        print(f"  {' '.join(p)} -> bi: {' '.join(g1)} | tri: {' '.join(g2)}")

    mejor_top1=top1_2 if top1_2>top1_1 else top1_1
    mejor=2 if top1_2>top1_1 else 1
    if mejor_top1>0.5:
        veredicto="FUNCIONAL: transicion explicita captura estructura suficiente para generar."
    elif mejor_top1>0.2:
        veredicto="PARCIAL: captura algo de estructura, pero generacion limitada."
    else:
        veredicto="NO FUNCIONAL: ni bigramas ni trigramas capturan secuencia."
    print(f"\nVeredicto final: {veredicto} (mejor orden={mejor}, top1={mejor_top1:.3f})")

    out=dict(experiment="v0.25_v13_transicion_explicita",
             results=dict(bigramas=dict(top1=top1_1, top5=top5_1),
                          trigramas=dict(top1=top1_2, top5=top5_2),
                          mejor_orden=mejor, mejor_top1=mejor_top1,
                          veredicto=veredicto,
                          generaciones_bigramas=gen1,
                          generaciones_trigramas=gen2))
    json.dump(out, open("results_v25_v13.json","w"), indent=2)
    print("-> results_v25_v13.json")

if __name__=="__main__":
    main()
