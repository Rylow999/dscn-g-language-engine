#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v11 — LOOP CONSERVADOR: actualiza contexto, no omega focal.
"""
import json, math, random, re
from collections import defaultdict

random.seed(0)

def norm(v): return math.sqrt(sum(x*x for x in v))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else dot(a,b)/(na*nb)

class SkipGram:
    def __init__(self, vocab, D=16, lr=0.05, window=5, neg_samples=5):
        self.D=D; self.lr=lr; self.window=window; self.neg_samples=neg_samples
        self.vocab=vocab
        self.emb={w:[random.gauss(0,0.1) for _ in range(D)] for w in vocab}
        self.ctx={w:[random.gauss(0,0.1) for _ in range(D)] for w in vocab}
    def train_pair(self, target, context, negative_samples):
        for d in range(self.D): self.emb[target][d]+=self.lr*(self.ctx[target][d]-self.emb[target][d])
    def sample_negative(self, rng, k):
        return rng.sample(self.vocab, min(k,len(self.vocab)))
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
                    neg=self.sample_negative(rng, self.neg_samples)
                    # positive
                    for d in range(self.D): self.emb[target][d]+=self.lr*(self.ctx[context][d]-self.emb[target][d])
                    # negatives
                    for ns in neg:
                        for d in range(self.D): self.emb[target][d]-=self.lr*self.ctx[ns][d]

class LinearSenseClassifier:
    def __init__(self, D, lr=0.05):
        self.D=D; self.lr=lr
        self.w=[random.gauss(0,0.1) for _ in range(D)]
        self.b=0.0
    def predict(self, x):
        return 1 if dot(self.w,x)+self.b > 0 else 0
    def update(self, x, label):
        yhat=1 if dot(self.w,x)+self.b > 0 else 0
        err=label-yhat
        for d in range(self.D): self.w[d]+=self.lr*err*x[d]
        self.b+=self.lr*err
    def fit(self, X, Y, epochs=10):
        rng=random.Random(2)
        for ep in range(epochs):
            idx=list(range(len(X))); rng.shuffle(idx)
            for i in idx: self.update(X[i], Y[i])

class SimpleTransformer:
    def __init__(self, vocab, D=16, lr=0.05):
        self.D=D; self.lr=lr; self.vocab=vocab
        self.emb={w:[random.gauss(0,0.1) for _ in range(D)] for w in vocab}
        self.omega={w:[random.gauss(0,0.1) for _ in range(D)] for w in vocab}
    def forward(self, ctx_words):
        omega_ctx=[0.0]*self.D; valid=0
        for w in ctx_words:
            if w in self.omega:
                for d in range(self.D): omega_ctx[d]+=self.omega[w][d]
                valid+=1
        if valid>0: omega_ctx=[x/valid for x in omega_ctx]
        return omega_ctx

class RootMemory:
    def __init__(self, transformer, D, beta_anchor=0.1, beta_repulse=0.02, theta=0.5, mm_alpha=0.8):
        self.transformer=transformer; self.D=D
        self.beta_anchor=beta_anchor; self.beta_repulse=beta_repulse; self.theta=theta
        self.mm_alpha=mm_alpha
        self.sub={}; self.foco={}
    def seed_word(self, word, seed_vecs):
        self.sub[word]=[list(seed_vecs[0]), list(seed_vecs[1])]
        self.foco[word]=0.0
    def update(self, word, context_words):
        if word not in self.sub: return
        A=self.sub[word][0]; B=self.sub[word][1]
        ctx=[0.0]*self.D; valid=0
        for w in context_words:
            if w in self.transformer.emb:
                for d in range(self.D): ctx[d]+=self.transformer.emb[w][d]
                valid+=1
        if valid>0: ctx=[x/valid for x in ctx]
        ca=cos(ctx,A); cb=cos(ctx,B)
        diff=abs(ca-cb)
        if diff > self.theta:
            if ca>=cb:
                for d in range(self.D): A[d]=self.mm_alpha*A[d]+(1-self.mm_alpha)*ctx[d]
                for d in range(self.D): B[d]-=self.beta_repulse*A[d]
            else:
                for d in range(self.D): B[d]=self.mm_alpha*B[d]+(1-self.mm_alpha)*ctx[d]
                for d in range(self.D): A[d]-=self.beta_repulse*B[d]
            for d in range(self.D):
                A[d]=max(-1.0,min(1.0,A[d])); B[d]=max(-1.0,min(1.0,B[d]))
            self.sub[word]=[A,B]
        self.foco[word]=diff

def build_corpus():
    word="banco"
    tplsA=[
        "fue al banco para dinero y pagar con tarjeta en mano",
        "el banco aprobo el interes sin plazo ni comision",
        "si tienes ahorro en el banco podras usar el cheque sin credito",
        "cerro su cuenta en el banco despues de retirar el saldo",
        "el banco publico ajusto la tasa de interes por la inflacion",
        "acredite el dinero en el banco para evitar el robo",
    ]
    tplsB=[
        "se tiro al banco del rio para pescar con su red",
        "el bote choco contra el banco de la orilla al remar",
        "amarraron la barca en el banco mientras el agua subia",
        "cerca del banco se pesco una trucha sobre la arena",
        "el puente esta sobre el banco para cruzarlo temprano",
        "bajamos por el banco del rio hasta la playa",
    ]
    seq=[]; meta=[]
    rng=random.Random(0)
    for _ in range(350):
        t=rng.choice(tplsA); seq.extend(t.split()); meta.extend(["A"]*len(t.split()))
        t=rng.choice(tplsB); seq.extend(t.split()); meta.extend(["B"]*len(t.split()))
    return seq, meta, word

def main():
    print("=== v0.25 v11 LOOP CONSERVADOR ===")
    seq,meta,word=build_corpus()
    vocab=sorted(set(seq))
    sg=SkipGram(vocab, D=16, lr=0.05, window=5, neg_samples=10)
    sg.fit(seq, epochs=10)
    W=8
    X=[]; Y=[]; word_positions=[]
    for i in range(len(seq)):
        if seq[i]==word:
            ctx=seq[max(0,i-W):i]
            vec=[0.0]*16; valid=0
            for w in ctx:
                if w in sg.emb:
                    for d in range(16): vec[d]+=sg.emb[w][d]
                    valid+=1
            if valid>0: vec=[x/valid for x in vec]
            X.append(vec); Y.append(1 if meta[i]=="A" else 0); word_positions.append(i)
    split=max(1, int(0.7*len(X)))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = Y[:split], Y[split:]
    test_positions = word_positions[split:]
    # baseline: clasificador lineal con skip-gram
    clf=LinearSenseClassifier(D=16, lr=0.05)
    clf.fit(X_train, y_train, epochs=20)
    correct_clf=sum(1 for x,y in zip(X_test,y_test) if clf.predict(x)==y)
    acc_clf=correct_clf/len(X_test) if len(X_test)>0 else 0.0
    print(f" acc_clf baseline={acc_clf:.3f} ({correct_clf}/{len(X_test)})")
    # loop conservador
    t=SimpleTransformer(vocab, D=16, lr=0.05)
    for w in vocab:
        if w in sg.emb: t.emb[w]=list(sg.emb[w])
        if w in sg.ctx: t.omega[w]=list(sg.ctx[w])
    r=RootMemory(t, D=16, beta_anchor=0.1, beta_repulse=0.02, theta=0.5, mm_alpha=0.8)
    r.seed_word(word,[[0.5]*16,[-0.5]*16])
    for ep_outer in range(3):
        for ep_inner in range(5):
            for i in range(W, len(seq)):
                if seq[i]!=word: continue
                ctx=seq[max(0,i-W):i]
                omega=t.forward(ctx)
                r.update(word, ctx)
                A,B=r.sub[word]; foco=r.foco[word]
                if foco > r.theta:
                    for cw in ctx:
                        if cw in t.emb:
                            for dim in range(16):
                                t.emb[cw][dim]+=0.01*foco*(A[dim]-B[dim])
                for dim in range(16):
                    t.omega[word][dim]=omega[dim]
        for i in range(W, len(seq)):
            ctx=seq[max(0,i-W):i]
            omega=t.forward(ctx)
            if seq[i] in t.emb:
                for dim in range(16):
                    t.emb[seq[i]][dim]+=0.02*(omega[dim]-t.emb[seq[i]][dim])
    X_loop=[]
    for pos in word_positions:
        ctx=seq[max(0,pos-W):pos]
        vec=[0.0]*16; vd=0
        for w in ctx:
            if w in t.emb:
                for d in range(16): vec[d]+=t.emb[w][d]
                vd+=1
        if vd>0: vec=[x/vd for x in vec]
        X_loop.append(vec)
    # evaluar con clasificador entrenado sobre embeddings finales del loop
    clf_loop=LinearSenseClassifier(D=16, lr=0.05)
    clf_loop.fit(X_loop[:split], y_train, epochs=20)
    correct_loop=sum(1 for x,y in zip(X_loop[split:],y_test) if clf_loop.predict(x)==y)
    acc_loop=correct_loop/len(X_test) if len(X_test)>0 else 0.0
    print(f" acc_loop={acc_loop:.3f} ({correct_loop}/{len(X_test)})")
    mejora=acc_loop-acc_clf
    print(f" mejora vs baseline={mejora:.3f}")
    if acc_loop>acc_clf+0.02:
        veredicto="LOOP MEJORA BASELINE: reglas conservadoras preservan señal."
    elif acc_loop<acc_clf-0.02:
        veredicto="LOOP EMPEORA BASELINE: sigue destruyendo."
    else:
        veredicto="LOOP INDISTINGUIBLE BASELINE."
    print(veredicto)
    json.dump(dict(experiment="v0.25_v11_loop_conservador", word=word,
                   results=dict(acc_clf=acc_clf, acc_loop=acc_loop, mejora=mejora,
                                 veredicto=veredicto)),
              open("results_v25_v11.json","w"), indent=2)
    print("-> results_v25_v11.json")
if __name__=="__main__":
    main()
