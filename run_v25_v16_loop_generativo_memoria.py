#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v16 — LOOP CERRADO GENERATIVO CON MEMORIA COMPETITIVA.
Objetivo: integrar el generador v14/v15 en un ciclo cerrado que preserve
el sentido activo durante generacion extendida.

Flujo:
  - contexto -> skip-gram -> omega contexto
  - clasificador lineal decide sentido activo (A/B)
  - memoria competitiva asigna foco al sentido ganador
  - generador bigrama condicionado al sentido produce siguientes tokens
  - nuevos tokens vuelven al contexto -> retroalimentacion

Metrica: coherencia de sentido a lo largo de la secuencia generada.
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

class LinearSenseClassifier:
    def __init__(self, D, lr=0.05):
        self.D=D; self.lr=lr
        self.w=[random.gauss(0,0.1) for _ in range(D)]
        self.b=0.0
    def predict(self, x):
        return 1 if sum(wi*xi for wi,xi in zip(self.w,x))+self.b > 0 else 0
    def update(self, x, label):
        yhat=1 if sum(wi*xi for wi,xi in zip(self.w,x))+self.b > 0 else 0
        err=label-yhat
        for d in range(self.D): self.w[d]+=self.lr*err*x[d]
        self.b+=self.lr*err
    def fit(self, X, Y, epochs=10):
        rng=random.Random(2)
        for ep in range(epochs):
            idx=list(range(len(X))); rng.shuffle(idx)
            for i in idx: self.update(X[i], Y[i])

class SenseMemory:
    def __init__(self, alpha=0.8, beta=0.05):
        self.alpha=alpha; self.beta=beta
        self.foco={"A":0.5, "B":0.5}
        self.hist={"A":[], "B":[]}
    def update(self, sense_pred):
        self.foco[sense_pred]+=self.beta
        total=sum(self.foco.values())
        if total>0:
            for k in self.foco: self.foco[k]/=total
        self.hist[sense_pred].append(1)
        self.hist["A" if sense_pred=="B" else "B"].append(0)
    def current(self):
        return max(self.foco, key=self.foco.get)

def main():
    print("=== v0.25 v16 LOOP GENERATIVO + MEMORIA COMPETITIVA ===")
    seq,meta,word=build_corpus(n_per_sense=350)
    vocab=sorted(set(seq))
    W=8

    # skip-gram
    print(" entrenando skip-gram...")
    sg=SkipGram(vocab, D=16, lr=0.05, window=5, neg_samples=10)
    sg.fit(seq, epochs=10)

    # dataset clasificador
    X=[]; Y=[]; positions=[]
    for i in range(len(seq)):
        if seq[i]==word:
            ctx=seq[max(0,i-W):i]
            vec=[0.0]*16; valid=0
            for w in ctx:
                if w in sg.emb:
                    for d in range(16): vec[d]+=sg.emb[w][d]
                    valid+=1
            if valid>0: vec=[x/valid for x in vec]
            X.append(vec); Y.append(1 if meta[i]=="A" else 0); positions.append(i)
    split=max(1, int(0.7*len(X)))
    X_train,X_test=X[:split],X[split:]
    y_train,y_test=Y[:split],Y[split:]
    test_positions=positions[split:]

    clf=LinearSenseClassifier(D=16, lr=0.05)
    clf.fit(X_train, y_train, epochs=20)
    acc_clf=sum(1 for x,y in zip(X_test,y_test) if clf.predict(x)==y)/len(X_test) if X_test else 0
    print(f" acc_clf={acc_clf:.3f}")

    # modelos bigrama por sentido
    sense_tokens={sense:[] for sense in ["A","B"]}
    for i in range(len(seq)): sense_tokens[meta[i]].append(seq[i])
    models={}
    for sense,tokens in sense_tokens.items():
        t=defaultdict(Counter)
        for w,wn in zip(tokens, tokens[1:]): t[w][wn]+=1
        models[sense]={w:{k:v/sum(c.values()) for k,v in c.items()} for w,c in t.items()}

    mem=SenseMemory(alpha=0.8, beta=0.05)
    # inicializar con verdad de test
    mem.foco={"A":0.5, "B":0.5}

    # generar secuencia desde contexto inicial
    ctx=seq[10:10+W]
    print(f"\n inicio: {' '.join(ctx)}")
    history=[]
    for step in range(20):
        vec=[0.0]*16; valid=0
        for w in ctx:
            if w in sg.emb:
                for d in range(16): vec[d]+=sg.emb[w][d]
                valid+=1
        if valid>0: vec=[x/valid for x in vec]
        sense_pred="A" if clf.predict(vec)==1 else "B"
        mem.update(sense_pred)
        sense_active=mem.current()
        probs=models[sense_active].get(ctx[-1], {})
        preds=sorted(probs, key=probs.get, reverse=True)[:5]
        nxt=preds[0] if preds else ""
        print(f"  paso {step}: cls={sense_pred} foco={mem.foco} activo={sense_active} -> '{nxt}'")
        ctx.append(nxt); ctx=ctx[-10:]
        history.append(sense_active)

    # metricas
    purity=Counter(history)
    total=sum(purity.values())
    dominant=max(purity, key=purity.get)
    coherence=purity[dominant]/total if total>0 else 0
    print(f"\n coherencia sentido activo: {coherence:.3f} ({dominant})")
    print(f" distribucion: {dict(purity)}")

    veredicto="NO FUNCIONAL"
    if coherence>0.7:
        veredicto="FUNCIONAL: memoria competitiva + generador por sentido mantiene coherencia."
    elif coherence>0.5:
        veredicto="PARCIAL: coherencia moderada, requiere ajuste."
    print(f"\n{veredicto}")
    json.dump(dict(experiment="v0.25_v16_loop_generativo_memoria", word=word,
                   results=dict(acc_clf=acc_clf, coherence=coherence,
                                dominant_sense=dominant, sense_distribution=dict(purity),
                                history=history, veredicto=veredicto)),
              open("results_v25_v16.json","w"), indent=2)
    print("-> results_v25_v16.json")

if __name__=="__main__":
    main()
