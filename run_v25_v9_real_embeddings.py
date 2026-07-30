#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corpus real + embeddings reales skip-gram para 'banco' en español.
Paso 1: descargar articulos Wikipedia sobre 'Banco (economia)' y 'Banco (geografia)'.
Paso 2: extraer oriciones con 'banco', etiquetar A/B segun fuente.
Paso 3: entrenar skip-gram simple sobre todo el corpus mixto.
Paso 4: ejecutar loop v8 con esos embeddings.
"""
import json, math, random, re, urllib.request, urllib.parse
from collections import defaultdict, Counter

random.seed(0)

def norm(v): return math.sqrt(sum(x*x for x in v))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else dot(a,b)/(na*nb)

# ---------- 1. descargar wikipedia ----------
def fetch_wikipedia(title):
    url="https://es.wikipedia.org/w/api.php"
    params=f"?action=query&prop=extracts&explaintext=true&titles={urllib.parse.quote(title)}&format=json"
    req=urllib.request.urlopen(url+params, timeout=30)
    data=json.loads(req.read().decode())
    pages=data["query"]["pages"]
    text=""
    for pid,p in pages.items():
        if "extract" in p:
            text=p["extract"]
    return text

# ---------- 2. limpiar + tokenizar ----------
def clean_tokenize(text):
    text=text.lower()
    text=re.sub(r"[^\w\sáéíóúüñ]"," ",text)
    return text.split()

# ---------- 3. skip-gram simple ----------
class SkipGram:
    def __init__(self, vocab, D=16, lr=0.05, window=5, neg_samples=5):
        self.D=D; self.lr=lr; self.window=window; self.neg_samples=neg_samples
        self.vocab=vocab
        self.emb={w:[random.gauss(0,0.1) for _ in range(D)] for w in vocab}
        self.ctx={w:[random.gauss(0,0.1) for _ in range(D)] for w in vocab}
    def train_pair(self, target, context, negative_samples):
        # positive
        for d in range(self.D):
            self.emb[target][d]+=self.lr*(self.ctx[context][d]-self.emb[target][d])
        # negatives
        for ns in negative_samples:
            for d in range(self.D):
                self.emb[target][d]-=self.lr*self.ctx[ns][d]
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

# ---------- 4. loop cerrado v8 ----------
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

def run_loop(seq, meta, word, W=8, epochs_outer=3, epochs_inner=5):
    vocab=sorted(set(seq))
    t=SimpleTransformer(vocab, D=16, lr=0.05)
    r=RootMemory(t, D=16, beta_anchor=0.2, beta_repulse=0.05, theta=0.8)
    seedA=[0.5]*16; seedB=[-0.5]*16
    r.seed_word(word,[seedA,seedB])
    for ep_outer in range(epochs_outer):
        for ep_inner in range(epochs_inner):
            for i in range(W, len(seq)):
                w=seq[i]
                if w!=word: continue
                ctx=seq[max(0,i-W):i]
                omega=t.forward(ctx)
                r.update(word, ctx)
                A,B=r.sub[word]
                foco=r.foco[word]
                for dim in range(16):
                    omega[dim]+=foco*(A[dim]-B[dim])
                t.omega[word]=list(omega)
        for i in range(W, len(seq)):
            ctx=seq[max(0,i-W):i]
            omega=t.forward(ctx)
            if seq[i] in t.emb:
                for dim in range(16):
                    t.emb[seq[i]][dim]+=0.05*(omega[dim]-t.emb[seq[i]][dim])
    correct=0; total=0
    for i in range(len(seq)):
        if seq[i]==word:
            ctx=seq[max(0,i-W):i]
            omega=t.forward(ctx)
            ca=cos(omega,r.sub[word][0]); cb=cos(omega,r.sub[word][1])
            pred="A" if ca>=cb else "B"
            if pred==meta[i]: correct+=1
            total+=1
    return correct/total if total>0 else 0.0

def main():
    print("=== Corpus real Wikipedia + embeddings skip-gram + loop v8 ===")
    # 1) corpus
    try:
        print(" descargando Wikipedia...")
        textoA=fetch_wikipedia("Banco (economia)")
        textoB=fetch_wikipedia("Banco (geografia)")
        tokensA=clean_tokenize(textoA)
        tokensB=clean_tokenize(textoB)
        seq=tokensA+tokensB
        meta=["A"]*len(tokensA)+["B"]*len(tokensB)
        word="banco"
        print(f" tokens A={len(tokensA)} B={len(tokensB)}")
    except Exception as e:
        print(f"ERROR descarga Wikipedia: {e}")
        print(" fallback: usar corpus sintetico ampliado")
        # fallback sintetico ampliado
        seq=[]; meta=[]; word="banco"
        POLY={
            "A": ["dinero","pagar","cuenta","ahorro","plata","banquero","interes","cheque","tarjeta","retiro"],
            "B": ["rio","agua","pez","orilla","puente","corriente","boga","remo","proa","popa"],
        }
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
        rng=random.Random(0)
        for _ in range(200):
            t=rng.choice(tplsA); toks=t.split(); seq.extend(toks); meta.extend(["A"]*len(toks))
            t=rng.choice(tplsB); toks=t.split(); seq.extend(toks); meta.extend(["B"]*len(toks))
        print(f" fallback seq={len(seq)}")
    # 2) entrenar skip-gram embeddings
    print(" entrenando skip-gram embeddings...")
    sg=SkipGram(sorted(set(seq)), D=16, lr=0.05, window=5, neg_samples=10)
    sg.fit(seq, epochs=10)
    # reemplazar embeddings del transformer por los skip-gram
    vocab=sorted(set(seq))
    t=SimpleTransformer(vocab, D=16, lr=0.05)
    for w in vocab:
        if w in sg.emb: t.emb[w]=list(sg.emb[w])
        if w in sg.ctx: t.omega[w]=list(sg.ctx[w])
    # 3) baseline sin loop
    correct_base=0; total=0
    W=8
    for i in range(len(seq)):
        if seq[i]==word:
            ctx=seq[max(0,i-W):i]
            omega=t.forward(ctx)
            ca=cos(omega,[0.5]*16); cb=cos(omega,[-0.5]*16)
            pred="A" if ca>=cb else "B"
            if pred==meta[i]: correct_base+=1
            total+=1
    baseline=correct_base/total if total>0 else 0.0
    print(f" baseline acc_decision={baseline:.3f} ({correct_base}/{total})")
    # 4) loop con embeddings reales
    loop=run_loop(seq,meta,word,W=8,epochs_outer=3,epochs_inner=5)
    print(f" loop acc_decision={loop:.3f}")
    mejora=loop-baseline
    print(f" mejora={mejora:.3f}")
    if loop>baseline+0.02:
        veredicto="LOOP MEJORA: embeddings reales ayudan a integrar."
    elif loop<baseline-0.02:
        veredicto="LOOP EMPEORA: embeddings reales no alcanzan."
    else:
        veredicto="LOOP INDISTINGUIBLE."
    print(veredicto)
    json.dump(dict(experiment="v0.25_v9_loop_embeddings_reales", word=word,
                   results=dict(baseline=baseline, loop=loop, mejora=mejora, veredicto=veredicto)),
              open("results_v25_v9.json","w"), indent=2)
    print("-> results_v25_v9.json")

if __name__=="__main__":
    main()
