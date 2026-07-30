#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v21 — CLASIFICADOR LINEAL + LOOP SOBRE EMBEDDINGS SKIP-GRAM.
"""
import json, math, random, re
from collections import defaultdict, Counter

random.seed(0)
WORD="banco"
W=8; D=32; EPOCHS_SG=10; EPOCHS_CLF=20; neg_samples=5

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
    def __init__(self, vocab, D, lr=0.05, window=5, neg_samples=5):
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
            for i in idx:
                self.update(X[i], Y[i])

class RootMemory:
    def __init__(self, transformer, D, lr=0.05, beta_anchor=0.2, beta_repulse=0.05, theta=0.8):
        self.transformer=transformer
        self.D=D; self.lr=lr; self.beta_anchor=beta_anchor; self.beta_repulse=beta_repulse; self.theta=theta
        self.omega=[random.gauss(0,0.1) for _ in range(D)]
        self.cotejo=[]; self.veredicto=None; self.diver=None
    def enraizar(self, A, B):
        anch_A=sum(wi*xi for wi,xi in zip(self.omega,A)); anch_B=sum(wi*xi for wi,xi in zip(self.omega,B))
        paso_A=[self.lr*(x-anch_A*wi) for wi,x in zip(self.omega,A)]
        paso_B=[self.lr*(x-anch_B*wi) for wi,x in zip(self.omega,B)]
        for d in range(self.D):
            self.omega[d]+=self.beta_anchor*paso_A[d]+self.beta_anchor*paso_B[d]
            sig=1.0 if (self.transformer.emb[self.transformer.current] if hasattr(self.transformer,"current") and self.transformer.current in self.transformer.emb else 0.0) > self.theta else -1.0
            ajuste=self.beta_repulse*(paso_B[d]-paso_A[d]) if sig>0 else -(paso_B[d]-paso_A[d])
            self.omega[d]+=ajuste
        dist_A=sum((a-b)**2 for a,b in zip(self.omega,A)); dist_B=sum((a-b)**2 for a,b in zip(self.omega,B))
        self.cotejo=[]; self.veredicto="EMPATE"; self.diver=dist_B-dist_A
        if dist_A < dist_B: self.veredicto="A"; self.cotejo=[dist_A,dist_B]
        else: self.veredicto="B"; self.cotejo=[dist_B,dist_A]

class SimpleTransformer:
    def __init__(self, vocab, D=16, lr=0.05):
        self.D=D; self.lr=lr; self.vocab=vocab
        self.emb={w:[random.gauss(0,0.1) for _ in range(D)] for w in vocab}
        self.current=None
    def contexto(self, seq):
        ctx=seq[-8:]
        vec=[0.0]*self.D; valid=0
        for w in ctx:
            if w in self.emb:
                for d in range(self.D): vec[d]+=self.emb[w][d]
                valid+=1
        if valid>0: vec=[x/valid for x in vec]
        self.current=ctx[-1] if ctx else None
        return vec

def acc(samples, predictor):
    correct=0
    for x,y in samples:
        if predictor(x)==y: correct+=1
    return correct/len(samples) if samples else 0

def eval_grupo(grupo, palabra):
    cnt=len(grupo)
    if cnt==0: return 0.0
    return sum(1 for x,y in grupo if x==palabra)/cnt

def main():
    print("=== v0.25 v21 CLASIFICADOR LINEAL + LOOP ===")
    seq,meta,word=build_corpus(n_per_sense=350, word=WORD)
    vocab=sorted(set(seq))
    # skip-gram
    print(" entrenando skip-gram...")
    sg=SkipGram(vocab, D=D, lr=0.05, window=5, neg_samples=neg_samples)
    sg.fit(seq, epochs=EPOCHS_SG)
    # dataset
    W=8; X=[]; Y=[]; labels_map={"A":1,"B":0}
    for i in range(len(seq)):
        if seq[i]==word:
            ctx=seq[max(0,i-W):i]
            vec=[0.0]*D; valid=0
            for w in ctx:
                if w in sg.emb:
                    for d in range(D): vec[d]+=sg.emb[w][d]
                    valid+=1
            if valid>0: vec=[x/valid for x in vec]
            X.append(vec); Y.append(labels_map[meta[i]] if meta[i] in labels_map else 1)
    split=max(1, int(0.7*len(X)))
    baseline_samples=[(x,y) for x,y in zip(X,Y)]
    # baseline: predictor ingenuo sobre embeddings
    def predictor_baseline(x):
        cosA=sum(a*b for a,b in zip(x,sg.emb.get(word,[0.0]*D)))  # naive
        return 1 if cosA>0 else 0
    baseline_acc=acc(baseline_samples, predictor_baseline)
    print(f" baseline_acc={baseline_acc:.3f}")
    # clasificador lineal
    clf=LinearSenseClassifier(D=D, lr=0.05)
    clf.fit(X[:split], Y[:split], epochs=EPOCHS_CLF)
    test_samples=[(x,y) for x,y in zip(X[split:], Y[split:])]
    clf_acc=acc(test_samples, clf.predict)
    print(f" clf_acc={clf_acc:.3f}")
    # loop
    print(" loop...")
    t=SimpleTransformer(vocab, D=16, lr=0.05)
    r=RootMemory(t, D=D, lr=0.05, beta_anchor=0.05, beta_repulse=0.02, theta=0.5)
    history=[]; correct_auto=0
    for x,y in baseline_samples[:200]:
        ctx_vec=x
        A=ctx_vec[:]; B=[-v for v in ctx_vec]
        r.enraizar(A, B)
        history.append(r.veredicto)
        if r.veredicto=="A" and y==1: correct_auto+=1
        elif r.veredicto=="B" and y==0: correct_auto+=1
    loop_acc=correct_auto/len(history) if history else 0
    print(f" loop_acc={loop_acc:.3f}")
    print([" ".join(["A"]*3), " ".join(["B"]*3)])
    # acc_gt real sobre word
    gt=[(x,y) for x,y in baseline_samples if x in X][:0]
    print(f" baseline_acc={baseline_acc:.3f} | clf_acc={clf_acc:.3f} | loop_acc={loop_acc:.3f}")
    out=dict(experiment="v0.25_v21_classifier_loop", word=WORD,
             baseline_acc=baseline_acc, clf_acc=clf_acc, loop_acc=loop_acc,
             veredicto="MEJORA LOOP" if loop_acc>baseline_acc else "NO MEJORA LOOP")
    json.dump(out, open("results_v25_v21.json","w"), indent=2)
    print("-> results_v25_v21.json")

if __name__=="__main__":
    main()
