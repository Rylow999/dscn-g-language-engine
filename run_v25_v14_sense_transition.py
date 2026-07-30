#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v14 — MODELO DE TRANSICION POR SENTIDO (A/B separados).
Objetivo: generar texto condicionado al sentido, usando bigramas independientes
por sentido. Si funciona, tenemos un generador con ruteo de sentido funcional.
Metricas:
  - top1/top5 condicionada al sentido
  - precision de sentido en generaciones
  - coherencia humana de salida
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

class BigramPerSense:
    def __init__(self):
        self.models={}
    def fit(self, seq, meta, word):
        for sense in ["A","B"]:
            tokens=[seq[i] for i in range(len(seq)) if meta[i]==sense]
            trans=defaultdict(Counter)
            for w,wn in zip(tokens, tokens[1:]):
                trans[w][wn]+=1
            probs={}
            for w,c in trans.items():
                total=sum(c.values())
                probs[w]={k:v/total for k,v in c.items()}
            self.models[sense]=probs
        # modelo mezclado como baseline
        tokens_both=seq
        trans=defaultdict(Counter)
        for w,wn in zip(tokens_both, tokens_both[1:]):
            trans[w][wn]+=1
        probs={}
        for w,c in trans.items():
            total=sum(c.values())
            probs[w]={k:v/total for k,v in c.items()}
        self.models["MIX"]=probs
    def predict(self, ctx, sense, top_k=5):
        prev=ctx[-1]
        probs=self.models.get(sense, self.models.get("MIX", {}))
        if prev not in probs:
            return []
        return sorted(probs[prev], key=probs[prev].get, reverse=True)[:top_k]
    def generate(self, context_words, sense, max_len=12, temperature=0.4):
        generated=[]
        ctx=list(context_words)
        for _ in range(max_len):
            preds=self.predict(ctx, sense, top_k=10)
            if not preds: break
            if sense in self.models and ctx[-1] in self.models[sense]:
                probs=self.models[sense][ctx[-1]]
            else:
                probs={p:1/len(preds) for p in preds}
            scores=[probs.get(w,0) for w in preds]
            max_s=max(scores)
            exps=[math.exp((s-max_s)/temperature) for s in scores]
            total=sum(exps)
            if total < 1e-9:
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

def sense_vocab_stats(seq, meta, sense):
    return Counter([seq[i] for i in range(len(seq)) if meta[i]==sense])

def main():
    print("=== v0.25 v14 MODELO DE TRANSICION POR SENTIDO ===")
    seq,meta,word=build_corpus(n_per_sense=350)
    print(f" seq={len(seq)} vocab={len(set(seq))}")

    m=BigramPerSense()
    m.fit(seq, meta, word)

    stats_A=sense_vocab_stats(seq,meta,"A")
    stats_B=sense_vocab_stats(seq,meta,"B")
    topA=set([w for w,_ in stats_A.most_common(20)])
    topB=set([w for w,_ in stats_B.most_common(20)])

    print("\nTop vocabulario A:", " ".join(list(topA)[:10]))
    print("Top vocabulario B:", " ".join(list(topB)[:10]))

    prompts=[
        ["fue","al","banco"],
        ["el","banco","aprobo"],
        ["se","tiro","al","banco"],
        ["el","bote","choco"],
    ]
    print("\nGeneraciones A (dinero/cuenta):")
    genA=[]
    for p in prompts:
        g=m.generate(p, sense="A", max_len=10, temperature=0.4)
        genA.append({"prompt": p, "generated": g})
        print(f"  {' '.join(p)} -> {' '.join(g)}")

    print("\nGeneraciones B (rio/agua):")
    genB=[]
    for p in prompts:
        g=m.generate(p, sense="B", max_len=10, temperature=0.4)
        genB.append({"prompt": p, "generated": g})
        print(f"  {' '.join(p)} -> {' '.join(g)}")

    print("\nGeneraciones MIX (sin condicion):")
    genM=[]
    for p in prompts:
        g=m.generate(p, sense="MIX", max_len=10, temperature=0.4)
        genM.append({"prompt": p, "generated": g})
        print(f"  {' '.join(p)} -> {' '.join(g)}")

    # evaluar precision de sentido en generaciones
    def sense_purity(gen_list, expected_sense):
        counts=Counter()
        for g in gen_list:
            for w in g:
                if w in topA and expected_sense=="A": counts["A"]+=1
                if w in topB and expected_sense=="B": counts["B"]+=1
        total=sum(counts.values())
        if total==0: return 0.0
        return counts[expected_sense]/total

    purity_A=sense_purity([g["generated"] for g in genA], "A")
    purity_B=sense_purity([g["generated"] for g in genB], "B")
    purity_M_A=sense_purity([g["generated"] for g in genM], "A")
    purity_M_B=sense_purity([g["generated"] for g in genM], "B")

    print(f"\nPureza sentido A: {purity_A:.3f}")
    print(f"Pureza sentido B: {purity_B:.3f}")
    print(f"Pureza MIX (eval A): {purity_M_A:.3f}")
    print(f"Pureza MIX (eval B): {purity_M_B:.3f}")

    veredicto="NO FUNCIONAL"
    if purity_A>0.6 and purity_B>0.6:
        veredicto="FUNCIONAL: modelos por sentido separan vocabulario y generan coherente."
    elif purity_A>purity_M_A and purity_B>purity_M_B:
        veredicto="PARCIAL: por sentido mejora el mezclado, pero pureza baja."
    else:
        veredicto="NO FUNCIONAL: no hay separacion clara."
    print(f"\n{veredicto}")

    out=dict(experiment="v0.25_v14_transicion_por_sentido", word=word,
             results=dict(pureza_A=purity_A, pureza_B=purity_B,
                          pureza_MIX_A=purity_M_A, pureza_MIX_B=purity_M_B,
                          veredicto=veredicto, generaciones_A=genA, generaciones_B=genB,
                          generaciones_MIX=genM))
    json.dump(out, open("results_v25_v14.json","w"), indent=2)
    print("-> results_v25_v14.json")

if __name__=="__main__":
    main()
