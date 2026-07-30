#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v10 — CLASIFICADOR LINEAL SUPERVISADO + LOOP SOBRE EMBEDDINGS SKIP-GRAM.
Reemplaza el predictor baseline ingenuo (coseno contra vectores fijos) por un
perceptron chico entrenado sobre contextos skip-gram. Si los embeddings separan
A/B, este baseline deberia ser alto incluso sin loop. Luego medimos loop sobre
ese baseline.
"""
import json, math, random, re, urllib.request, urllib.parse
from collections import defaultdict, Counter

random.seed(0)

def norm(v): return math.sqrt(sum(x*x for x in v))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else dot(a,b)/(na*nb)

# ---------- wikipedia ----------
def fetch_wikipedia(title):
    url="https://es.wikipedia.org/w/api.php"
    params=f"?action=query&prop=extracts&explaintext=true&titles={urllib.parse.quote(title)}&format=json"
    req=urllib.request.urlopen(url+params, timeout=30)
    data=json.loads(req.read().decode())
    pages=data["query"]["pages"]; text=""
    for pid,p in pages.items():
        if "extract" in p: text=p["extract"]
    return text

def clean_tokenize(text):
    text=text.lower(); text=re.sub(r"[^\w\sáéíóúüñ]"," ",text)
    return text.split()

# ---------- skip-gram ----------
class SkipGram:
    def __init__(self, vocab, D=16, lr=0.05, window=5, neg_samples=5):
        self.D=D; self.lr=lr; self.window=window; self.neg_samples=neg_samples
        self.vocab=vocab
        self.emb={w:[random.gauss(0,0.1) for _ in range(D)] for w in vocab}
        self.ctx={w:[random.gauss(0,0.1) for _ in range(D)] for w in vocab}
    def train_pair(self, target, context, negative_samples):
        for d in range(self.D): self.emb[target][d]+=self.lr*(self.ctx[context][d]-self.emb[target][d])
        for ns in negative_samples:
            for d in range(self.D): self.emb[target][d]-=self.lr*self.ctx[ns][d]
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
                    self.train_pair(target, context, neg)

# ---------- clasificador lineal ----------
class LinearSenseClassifier:
    def __init__(self, D, lr=0.05):
        self.D=D; self.lr=lr
        self.w=[random.gauss(0,0.1) for _ in range(D)]
        self.b=0.0
    def predict(self, x):
        return 1 if dot(self.w,x) + self.b > 0 else 0  # 1=A, 0=B
    def update(self, x, label):
        # label: 1=A, 0=B
        yhat=1 if dot(self.w,x)+self.b > 0 else 0
        err=label-yhat
        for d in range(self.D): self.w[d]+=self.lr*err*x[d]
        self.b+=self.lr*err
    def fit(self, X, Y, epochs=10):
        for ep in range(epochs):
            rng=random.Random(2)
            idx=list(range(len(X))); rng.shuffle(idx)
            for i in idx:
                self.update(X[i], Y[i])

# ---------- loop ----------
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
    def __init__(self, transformer, D, beta_anchor=0.2, beta_repulse=0.05, theta=0.8):
        self.transformer=transformer; self.D=D
        self.beta_anchor=beta_anchor; self.beta_repulse=beta_repulse; self.theta=theta
        self.sub={}; self.foco={}
    def seed_word(self, word, seed_vecs):
        self.sub[word]=[list(seed_vecs[0]), list(seed_vecs[1])]
        self.foco[word]=0.0
    def update(self, word, context_words):
        A=self.sub[word][0]; B=self.sub[word][1]
        ctx=[0.0]*self.D; valid=0
        for w in context_words:
            if w in self.transformer.emb:
                for d in range(self.D): ctx[d]+=self.transformer.emb[w][d]
                valid+=1
        if valid>0: ctx=[x/valid for x in ctx]
        ca=cos(ctx,A); cb=cos(ctx,B)
        if ca>=cb:
            for d in range(self.D): A[d]+=self.beta_anchor*(ctx[d]-A[d])
            if ca<self.beta_repulse or ca<self.theta:
                for d in range(self.D): B[d]-=self.beta_repulse*A[d]
        else:
            for d in range(self.D): B[d]+=self.beta_anchor*(ctx[d]-B[d])
            if cb<self.beta_repulse or cb<self.theta:
                for d in range(self.D): A[d]-=self.beta_repulse*B[d]
        for d in range(self.D):
            A[d]=max(-1.0,min(1.0,A[d])); B[d]=max(-1.0,min(1.0,B[d]))
        self.sub[word]=[A,B]
        self.foco[word]=abs(ca-cb)

# ---------- experimento ----------
def main():
    print("=== v0.25 v10 CLASIFICADOR LINEAL + LOOP sobre skip-gram ===")
    # corpus
    try:
        print(" descargando Wikipedia...")
        textoA=fetch_wikipedia("Banco (economia)")
        textoB=fetch_wikipedia("Banco (geografia)")
        tokensA=clean_tokenize(textoA); tokensB=clean_tokenize(textoB)
        if len(tokensA)<20 or len(tokensB)<20:
            raise ValueError("corpus Wikipedia muy corto")
        seq=tokensA+tokensB
        meta=["A"]*len(tokensA)+["B"]*len(tokensB)
        word="banco"
        print(f" tokens A={len(tokensA)} B={len(tokensB)}")
    except Exception as e:
        print(f"ERROR Wikipedia ({e}): usando corpus sintetico ampliado")
        seq=[]; meta=[]; word="banco"
        tplsA=[
            "fue al banco para dinero y pagar con tarjeta en mano",
            "el banco aprobo el interes sin plazo ni comision",
            "si tienes ahorro en el banco podras usar el cheque sin credito",
            "cerro su cuenta en el banco despues de retirar el saldo",
            "el banco publico ajusto la tasa de interes por la inflacion",
            "acredite el dinero en el banco para evitar el robo",
            "la entidad bancaria aumento el limite de la tarjeta",
            "cobraron una comision en el banco por el mantenimiento",
            "el cliente pidio un prestamo en el banco oficial",
            "el banco central subio la tasa de ahorro again",
        ]
        tplsB=[
            "se tiro al banco del rio para pescar con su red",
            "el bote choco contra el banco de la orilla al remar",
            "amarraron la barca en el banco mientras el agua subia",
            "cerca del banco se pesco una trucha sobre la arena",
            "el puente esta sobre el banco para cruzarlo temprano",
            "bajamos por el banco del rio hasta la playa",
            "la corriente arrastro el tronco hacia el banco",
            "pescamos en el banco del rio con caña y carnada",
            "el nivel del rio cubrio el banco despues de la lluvia",
            "caminamos por el banco del lago al atardecer",
        ]
        rng=random.Random(0)
        for _ in range(250):
            t=rng.choice(tplsA); toks=t.split(); seq.extend(toks); meta.extend(["A"]*len(toks))
            t=rng.choice(tplsB); toks=t.split(); seq.extend(toks); meta.extend(["B"]*len(toks))
        print(f" fallback seq={len(seq)}")

    # skip-gram
    print(" entrenando skip-gram...")
    vocab=sorted(set(seq))
    sg=SkipGram(vocab, D=16, lr=0.05, window=5, neg_samples=10)
    sg.fit(seq, epochs=10)

    # contextos para clasificador
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

    # split train/test
    split=max(1, int(0.7*len(X)))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = Y[:split], Y[split:]
    test_positions = word_positions[split:]
    print(f" ejemplos A/B: train={len(X_train)} test={len(X_test)}")

    # baseline 1: clasificador lineal supervisado
    clf=LinearSenseClassifier(D=16, lr=0.05)
    clf.fit(X_train, y_train, epochs=20)
    correct_clf=0
    for x,y in zip(X_test, y_test):
        if clf.predict(x)==y: correct_clf+=1
    acc_clf=correct_clf/len(X_test) if len(X_test)>0 else 0.0
    print(f" acc_clf baseline={acc_clf:.3f} ({correct_clf}/{len(X_test)})")

    # baseline 2: loop con embeddings skip-gram en transformer
    t=SimpleTransformer(vocab, D=16, lr=0.05)
    for w in vocab:
        if w in sg.emb: t.emb[w]=list(sg.emb[w])
        if w in sg.ctx: t.omega[w]=list(sg.ctx[w])
    r=RootMemory(t, D=16, beta_anchor=0.2, beta_repulse=0.05, theta=0.8)
    seedA=[0.5]*16; seedB=[-0.5]*16
    r.seed_word(word,[seedA,seedB])
    for ep_outer in range(3):
        for ep_inner in range(5):
            for i in range(W, len(seq)):
                if seq[i]!=word: continue
                ctx=seq[max(0,i-W):i]
                omega=t.forward(ctx)
                r.update(word, ctx)
                A,B=r.sub[word]; foco=r.foco[word]
                for dim in range(16): omega[dim]+=foco*(A[dim]-B[dim])
                t.omega[word]=list(omega)
        for i in range(W, len(seq)):
            ctx=seq[max(0,i-W):i]
            omega=t.forward(ctx)
            if seq[i] in t.emb:
                for dim in range(16): t.emb[seq[i]][dim]+=0.05*(omega[dim]-t.emb[seq[i]][dim])
    correct_loop=0; total=0
    for pos,label in zip(test_positions, y_test):
        ctx=seq[max(0,pos-W):pos]
        omega=t.forward(ctx)
        ca=cos(omega,r.sub[word][0]); cb=cos(omega,r.sub[word][1])
        pred=1 if ca>=cb else 0
        if pred==label: correct_loop+=1
        total+=1
    acc_loop=correct_loop/total if total>0 else 0.0
    print(f" acc_loop={acc_loop:.3f} ({correct_loop}/{total})")
    mejora=acc_loop-acc_clf
    print(f" mejora vs clasificador={mejora:.3f}")
    if acc_loop>acc_clf+0.02:
        veredicto="LOOP MEJORA CLASIFICADOR: agregar root/memoria ayuda."
    elif acc_loop<acc_clf-0.02:
        veredicto="LOOP EMPEORA CLASIFICADOR: el ciclo introduce ruido."
    else:
        veredicto="LOOP INDISTINGUIBLE del clasificador."
    print(veredicto)
    json.dump(dict(experiment="v0.25_v10_clasificador_lineal", word=word,
                   results=dict(acc_clf=acc_clf, acc_loop=acc_loop, mejora=mejora,
                                 baseline_clf=f"{correct_clf}/{len(X_test)}",
                                 loop_correct=f"{correct_loop}/{total}",
                                 veredicto=veredicto)),
              open("results_v25_v10.json","w"), indent=2)
    print("-> results_v25_v10.json")
if __name__=="__main__":
    main()
