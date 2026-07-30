#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v15 — LOOP GENERATIVO POR SENTIDO (transicion A/B + clasificador).
Objetivo: integrar el generador v14 en un ciclo cerrado.
Flujo:
  1. Contexto -> embeddings skip-gram
  2. Clasificador lineal predice sentido activo (A/B)
  3. Generador bigrama condicionado al sentido genera siguientes tokens
  4. Esos tokens vuelven al contexto y el ciclo se repite
Metrica: coherencia de secuencia + precision de sentido generado.
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
    def generate(self, context_words, sense, max_len=10):
        out=[]
        ctx=list(context_words)
        for _ in range(max_len):
            preds=self.predict(ctx, top_k=10)
            if not preds: break
            scores=[self.probs[ctx[-1]].get(w,0) for w in preds]
            max_s=max(scores)
            exps=[math.exp((s-max_s)/0.4) for s in scores]
            total=sum(exps)
            if total < 1e-9: nxt=preds[0]
            else:
                probs_norm=[e/total for e in exps]
                r=random.random(); cum=0; nxt=preds[0]
                for w,p in zip(preds, probs_norm):
                    cum+=p
                    if r<=cum:
                        nxt=w
                        break
            out.append(nxt)
            ctx.append(nxt)
            ctx=ctx[-10:]
        return out

def main():
    print("=== v0.25 v15 LOOP GENERATIVO POR SENTIDO ===")
    seq,meta,word=build_corpus(n_per_sense=350)
    vocab=sorted(set(seq))
    W=8

    # 1. skip-gram
    print(" entrenando skip-gram...")
    sg=SkipGram(vocab, D=16, lr=0.05, window=5, neg_samples=10)
    sg.fit(seq, epochs=10)

    # 2. dataset clasificador
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

    # 3. generar secuencia completa con clasificador + bigramas por sentido
    # observamos el sentido de cada ocurrencia y generamos siguientes tokens
    true_sense=[]; pred_sense=[]; gen_tokens=[]
    for pos,label in zip(test_positions, y_test):
        ctx=seq[max(0,pos-W):pos]
        vec=[0.0]*16; valid=0
        for w in ctx:
            if w in sg.emb:
                for d in range(16): vec[d]+=sg.emb[w][d]
                valid+=1
        if valid>0: vec=[x/valid for x in vec]
        sense_pred="A" if clf.predict(vec)==1 else "B"
        true_sense.append("A" if label==1 else "B")
        pred_sense.append(sense_pred)

    # 4. metricas
    acc_sense=sum(1 for t,p in zip(true_sense,pred_sense) if t==p)/len(true_sense) if true_sense else 0
    print(f" acc_sense={acc_sense:.3f}")

    # 5. ejemplo generativo completo sobre secuencia continua
    start=10
    ctx_window=seq[start-W:start]
    for step in range(12):
        vec=[0.0]*16; valid=0
        for w in ctx_window:
            if w in sg.emb:
                for d in range(16): vec[d]+=sg.emb[w][d]
                valid+=1
        if valid>0: vec=[x/valid for x in vec]
        sense="A" if clf.predict(vec)==1 else "B"
        # bigrama por sentido
        transitions=defaultdict(Counter)
        sense_tokens=[seq[i] for i in range(len(seq)) if meta[i]==sense]
        for a,b in zip(sense_tokens, sense_tokens[1:]):
            transitions[a][b]+=1
        prev=ctx_window[-1]
        preds=sorted(transitions[prev], key=transitions[prev].get, reverse=True)[:5]
        if not preds:
            break
        nxt=preds[0]
        print(f"  paso {step}: sentido={sense}, generado='{nxt}'")
        ctx_window.append(nxt)
        ctx_window=ctx_window[-10:]

    print(f"\nVeredicto: acc_sense={acc_sense:.3f}. Si >=0.6, loop con sentido funcional.")
    json.dump(dict(experiment="v0.25_v15_loop_generativo_sentido", word=word,
                   results=dict(acc_clf=acc_clf, acc_sense=acc_sense, true_sense=true_sense[:20],
                                pred_sense=pred_sense[:20])),
              open("results_v25_v15.json","w"), indent=2)
    print("-> results_v25_v15.json")

if __name__=="__main__":
    main()
