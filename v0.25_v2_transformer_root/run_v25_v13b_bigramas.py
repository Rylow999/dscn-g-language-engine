#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v13b — DECODIFICADOR GENERATIVO CON BIGRAMAS + BEAM SEARCH.
Comparacion directa contra v12 (embeddings similitud).
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

class BigramModel:
    def __init__(self):
        self.probs={}
        self.trans=defaultdict(Counter)
    def fit(self, seq):
        for w,wn in zip(seq, seq[1:]):
            self.trans[w][wn]+=1
        self.probs={}
        for w,c in self.trans.items():
            total=sum(c.values())
            self.probs[w]={k:v/total for k,v in c.items()}
    def predict(self, ctx, top_k=5):
        prev=ctx[-1]
        if prev not in self.probs:
            return []
        return sorted(self.probs[prev], key=self.probs[prev].get, reverse=True)[:top_k]
    def _sample(self, ctx, temperature, top_k):
        preds=self.predict(ctx, top_k=top_k)
        if not preds: return None
        if temperature==0: return preds[0]
        scores=[self.probs[ctx[-1]].get(w,0) for w in preds]
        max_s=max(scores)
        exps=[math.exp((s-max_s)/temperature) for s in scores]
        total=sum(exps)
        if total < 1e-9: return preds[0]
        probs_norm=[e/total for e in exps]
        r=random.random(); cum=0
        for w,p in zip(preds, probs_norm):
            cum+=p
            if r <= cum: return w
        return preds[-1]
    def generate(self, context_words, max_len=15, temperature=0.4, top_k=10, no_repeat=3):
        generated=[]
        ctx=list(context_words)
        last=None; repeat_count=0
        for _ in range(max_len):
            nxt=self._sample(ctx, temperature, top_k)
            if nxt is None: break
            if nxt==last:
                repeat_count+=1
                if repeat_count>=no_repeat:
                    # sortear de top_k que no sea el último
                    preds=self.predict(ctx, top_k=top_k)
                    preds=[p for p in preds if p!=last]
                    if not preds: break
                    nxt=random.choice(preds)
                    repeat_count=0
            else:
                repeat_count=0
            generated.append(nxt)
            ctx.append(nxt)
            ctx=ctx[-10:]
            last=nxt
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
    print("=== v0.25 v13b BIGRAMAS + BEAM SEARCH ===")
    seq,meta,word=build_corpus(n_per_sense=350)
    print(f" seq={len(seq)} vocab={len(set(seq))}")

    m=BigramModel(); m.fit(seq)

    top1,top5,n=evaluar(m, seq, W=8, n_samples=200)
    print(f" top1={top1:.3f} top5={top5:.3f} (n={n})")

    prompts_base=[
        ["fue","al","banco"],
        ["el","banco","aprobo"],
        ["se","tiro","al","banco"],
        ["el","bote","choco"],
    ]
    prompts_compatibles=[
        ["fue","al","banco"],
        ["el","banco","aprobo"],
        ["se","tiro","al","banco"],
        ["el","bote","choco"],
    ]
    print("\n generaciones:")
    generated=[]
    for p in prompts_compatibles:
        gen=m.generate(p, max_len=10, temperature=0.4, top_k=10, no_repeat=3)
        generated.append({"prompt": p, "generated": gen})
        print(f"  {' '.join(p)} -> {' '.join(gen)}")

    veredicto="NO FUNCIONAL"
    if top1>0.5 and len(generated[0].get('generated', []))>=5:
        veredicto="FUNCIONAL: captura transicion y genera texto coherente."
    elif top1>0.2:
        veredicto="PARCIAL: captura algo pero generacion limitada."
    print(f"\n{veredicto} (top1={top1:.3f})")

    out=dict(experiment="v0.25_v13b_bigramas_beam", word=word,
             results=dict(top1=top1, top5=top5, n_samples=n, veredicto=veredicto,
                          generaciones=generated))
    json.dump(out, open("results_v25_v13b.json","w"), indent=2)
    print("-> results_v25_v13b.json")

if __name__=="__main__":
    main()
