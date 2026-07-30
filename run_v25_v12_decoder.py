#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v12 — DECODIFICADOR GENERATIVO SOBRE EMBEDDINGS SKIP-GRAM.
Objetivo: generar texto coherente desde embeddings reales.
Metodo:
  - Entrenar skip-gram sobre corpus sintetico ampliado A/B con ground truth.
  - Implementar decoder: dado un vector omega (contexto), predecir siguiente
    palabra por similitud coseno contra embeddings.
  - Medir: precision top-1, top-5, y coherencia de secuencias generadas.
  - Iterar hasta que el output sea comprensible.
"""
import json, math, random, re
from collections import defaultdict, Counter

random.seed(0)

def norm(v): return math.sqrt(sum(x*x for x in v))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else dot(a,b)/(na*nb)

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

class SkipGram:
    def __init__(self, vocab, D=16, lr=0.05, window=5, neg_samples=5):
        self.D=D; self.lr=lr; self.window=window; self.neg_samples=neg_samples
        self.vocab=vocab
        self.emb={w:[random.gauss(0,0.1) for _ in range(D)] for w in vocab}
        self.ctx={w:[random.gauss(0,0.1) for _ in range(D)] for w in vocab}
    def fit(self, tokens, epochs=10):
        rng=random.Random(1)
        for ep in range(epochs):
            for i in range(len(tokens)):
                target=tokens[i]
                if target not in self.emb: continue
                start=max(0,i-self.window); end=min(len(tokens),i+self.window+1)
                for j in range(start,end):
                    if j==i: continue
                    context=tokens[j]
                    if context not in self.ctx or context==target: continue
                    neg=rng.sample(self.vocab, min(self.neg_samples,len(self.vocab)))
                    for d in range(self.D): self.emb[target][d]+=self.lr*(self.ctx[context][d]-self.emb[target][d])
                    for ns in neg:
                        for d in range(self.D): self.emb[target][d]-=self.lr*self.ctx[ns][d]

class Decoder:
    def __init__(self, vocab, sg):
        self.vocab=vocab; self.sg=sg
    def predict(self, omega, top_k=5):
        if not omega or len(omega)!=self.sg.D:
            return []
        scores={}
        for w in self.vocab:
            s=cos(omega, self.sg.emb[w])
            scores[w]=s
        return sorted(scores, key=scores.get, reverse=True)[:top_k]
    def generate(self, context_words, max_len=15, temperature=1.0):
        # Generar secuencia desde contexto inicial
        generated=[]
        ctx=list(context_words)
        for _ in range(max_len):
            omega=[0.0]*self.sg.D; valid=0
            for w in ctx:
                if w in self.sg.emb:
                    for d in range(self.sg.D): omega[d]+=self.sg.emb[w][d]
                    valid+=1
            if valid>0: omega=[x/valid for x in omega]
            candidates=self.predict(omega, top_k=5)
            if not candidates: break
            # muestrear con temperatura
            if temperature==0:
                next_word=candidates[0]
            else:
                scores=[cos(omega, self.sg.emb[w]) for w in candidates]
                # softmax simple
                max_s=max(scores)
                exps=[math.exp((s-max_s)/temperature) for s in scores]
                probs=[e/sum(exps) for e in exps]
                r=random.random()
                cum=0; next_word=candidates[0]
                for w,p in zip(candidates, probs):
                    cum+=p
                    if r<=cum:
                        next_word=w
                        break
            generated.append(next_word)
            ctx.append(next_word)
            ctx=ctx[-10:]  # ventana deslizante
        return generated

def evaluate_decoder(decoder, seq, W=8, n_samples=100):
    # top-1 / top-5 precision en siguiente token
    correct1=0; correct5=0; total=0
    rng=random.Random(3)
    indices=rng.sample(range(W, len(seq)), min(n_samples, len(seq)-W))
    for i in indices:
        ctx=seq[max(0,i-W):i]
        omega=[0.0]*16; valid=0
        for w in ctx:
            if w in decoder.sg.emb:
                for d in range(16): omega[d]+=decoder.sg.emb[w][d]
                valid+=1
        if valid>0: omega=[x/valid for x in omega]
        preds=decoder.predict(omega, top_k=5)
        target=seq[i]
        if preds and preds[0]==target: correct1+=1
        if target in preds: correct5+=1
        total+=1
    top1=correct1/total if total>0 else 0.0
    top5=correct5/total if total>0 else 0.0
    return top1, top5, total

def evaluate_sequences(decoder, test_prompts):
    # generar texto desde prompts
    outputs=[]
    for prompt in test_prompts:
        gen=decoder.generate(prompt, max_len=10, temperature=0.5)
        outputs.append({"prompt": prompt, "generated": gen})
    return outputs

def main():
    print("=== v0.25 v12 DECODIFICADOR GENERATIVO ===")
    seq,meta,word=build_corpus(n_per_sense=350)
    vocab=sorted(set(seq))
    print(f" seq={len(seq)} vocab={len(vocab)}")

    # skip-gram
    print(" entrenando skip-gram...")
    sg=SkipGram(vocab, D=16, lr=0.05, window=5, neg_samples=10)
    sg.fit(seq, epochs=10)

    # decoder
    d=Decoder(vocab, sg)

    # evaluar precision top-1/top-5
    top1, top5, n = evaluate_decoder(d, seq, W=8, n_samples=200)
    print(f" top1={top1:.3f} top5={top5:.3f} (n={n})")

    # generar secuencias de prueba
    prompts=[
        ["fue","al","banco"],
        ["el","banco","aprobo"],
        ["se","tiro","al","banco"],
        ["el","bote","choco"],
        ["la","llave","de","la"],
    ]
    outputs=evaluate_sequences(d, prompts)
    print("\n generaciones:")
    for o in outputs:
        print(f"  {' '.join(o['prompt'])} -> {' '.join(o['generated'])}")

    # guardar resultados
    out=dict(
        experiment="v0.25_v12_decoder_generativo",
        results=dict(top1=top1, top5=top5, n_samples=n, outputs=outputs)
    )
    json.dump(out, open("results_v25_v12.json","w"), indent=2)
    print("\n-> results_v25_v12.json")

    # criterio de exito
    if top1 > 0.3:
        veredicto="FUNCIONAL: precision top1 suficiente para generar texto coherente."
    elif top5 > 0.7:
        veredicto="PARCIAL: top5 decente, top1 debil. Coherencia limitada."
    else:
        veredicto="NO FUNCIONAL: decodificador no captura estructura del corpus."
    print(veredicto)
    out["veredicto"]=veredicto
    json.dump(out, open("results_v25_v12.json","w"), indent=2)

if __name__=="__main__":
    main()
