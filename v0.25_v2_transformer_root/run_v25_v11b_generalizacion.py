#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v11b — GENERALIZACION: loop conservador sobre otra palabra.
Objetivo: confirmar que v11 no es un artefacto de 'banco'.
"""
import json, math, random, re
from collections import defaultdict

random.seed(0)

def norm(v): return math.sqrt(sum(x*x for x in v))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else dot(a,b)/(na*nb)

POLYSEMY = {
    "llave": {
        "A": ["puerta","entrar","casa","cerradura","hogar","domicilio","acceso","cerrar","abrir","seguridad"],
        "B": ["musica","nota","piano","acorde","melodia","tono","tecla","sinfonía","solfeo","armonia"],
        "templates_A": [
            "perdio la llave de la puerta al entrar a la casa",
            "la cerradura necesita una llave para abrir la puerta",
            "dejo la llave en la cerradura de la casa al salir",
            "el cerrajero abrio la puerta sin la llave",
            "la puerta se abre solo con la llave correcta",
            "no encontre la llave de la casa después de entrar",
        ],
        "templates_B": [
            "toco la llave del piano con suavidad",
            "la llave menor suena triste en el piano",
            "aprendio la llave de sol en solfeo",
            "cada llave del piano produce una nota distinta",
            "la llave de fa esta en la tercera linea",
            "interpretaron la llave mayor con alegria",
        ],
    },
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
    def __init__(self, transformer, D):
        self.transformer=transformer; self.D=D
        self.beta_anchor=0.1; self.beta_repulse=0.02; self.theta=0.5; self.mm_alpha=0.8
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

def run_loop(seq,meta,word,W=8,epochs_outer=3,epochs_inner=5):
    vocab=sorted(set(seq))
    sg=SkipGram(vocab, D=16, lr=0.05, window=5, neg_samples=10)
    sg.fit(seq, epochs=10)
    t=SimpleTransformer(vocab, D=16, lr=0.05)
    for w in vocab:
        if w in sg.emb: t.emb[w]=list(sg.emb[w])
        if w in sg.ctx: t.omega[w]=list(sg.ctx[w])
    r=RootMemory(t, D=16)
    r.seed_word(word,[[0.5]*16,[-0.5]*16])
    for _ in range(epochs_outer):
        for _ in range(epochs_inner):
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
    return t, r

def evaluate(seq,meta,word,t,r,W=8):
    correct=0; total=0
    for i in range(len(seq)):
        if seq[i]==word:
            ctx=seq[max(0,i-W):i]
            vec=[0.0]*16; valid=0
            for w in ctx:
                if w in t.emb:
                    for d in range(16): vec[d]+=t.emb[w][d]
                    valid+=1
            if valid>0: vec=[x/valid for x in vec]
            ca=cos(vec,r.sub[word][0]); cb=cos(vec,r.sub[word][1])
            pred="A" if ca>=cb else "B"
            if pred==meta[i]: correct+=1
            total+=1
    return correct/total if total>0 else 0.0, total

def main():
    print("=== v0.25 v11b GENERALIZACION ===")
    words=["llave","banco"]
    out={}
    for word in words:
        print(f"\n--- LOOP conservador sobre '{word}' ---")
        seq,meta,word=build_corpus(n_per_sense=350, word=word)
        vocab=sorted(set(seq))
        X=[]; Y=[]; positions=[]
        W=8
        for i in range(len(seq)):
            if seq[i]==word:
                ctx=seq[max(0,i-W):i]
                vec=[0.0]*16; valid=0
                for w in ctx:
                    if w in seq:  # placeholder, skip-gram depois
                        valid+=1
                # llenar X/Y despues del loop
                positions.append(i)
                Y.append(1 if meta[i]=="A" else 0)
        # baseline y loop con skip-gram
        sg=SkipGram(vocab, D=16, lr=0.05, window=5, neg_samples=10)
        sg.fit(seq, epochs=10)
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
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = Y[:split], Y[split:]
        clf=LinearSenseClassifier(D=16, lr=0.05)
        clf.fit(X_train, y_train, epochs=20)
        correct_clf=sum(1 for x,y in zip(X_test,y_test) if clf.predict(x)==y)
        acc_clf=correct_clf/len(X_test) if len(X_test)>0 else 0.0
        print(f" acc_clf baseline={acc_clf:.3f}")
        # loop
        t,r=run_loop(seq,meta,word,W=8,epochs_outer=3,epochs_inner=5)
        acc_loop,total_loop=evaluate(seq,meta,word,t,r,W=8)
        print(f" acc_loop={acc_loop:.3f} (n={total_loop})")
        mejora=acc_loop-acc_clf
        out[word]=dict(acc_clf=acc_clf, acc_loop=acc_loop, mejora=mejora)
    print("\nRESULTADOS:", json.dumps(out, indent=2))
    json.dump(dict(experiment="v0.25_v11b_generalizacion", results=out), open("results_v25_v11b.json","w"), indent=2)
    print("-> results_v25_v11b.json")
if __name__=="__main__":
    main()
