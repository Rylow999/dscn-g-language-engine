#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v8b — LOOP SIN DECODIFICADOR: solo transformer + root + memoria.
Objetivo: aislar si el decodificador era el responsable del colapso en v0.25 v8.
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
            "fue al banco para {A1} y {A2} con {A3} en mano",
            "el banco aprobo el {A2} sin {A3} ni {A4} de por medio",
            "si tienes {A1} en el banco podras usar el {A2} sin {A3}",
            "cerro su {A1} en el banco despues de {A2} el {A3}",
        ],
        "templates_B": [
            "se tiro al banco del {B1} para {B2} con su {B3}",
            "el {B3} choco contra el banco de la {B1} al {B2}",
            "amarraron la barca en el banco mientras el {B1} {B2}",
            "cerca del banco se {B2} una {B3} sobre la {B1}",
        ],
    },
}
def fill_template(tpl, sense, word):
    def rep(m):
        key=m.group(0).strip("{}")
        idx=int(key[1:])-1
        return POLYSEMY[word][sense][idx % len(POLYSEMY[word][sense])]
    return re.sub(r'\{[AB]\d+\}', rep, tpl)

def build_corpus(n_per_sense=60, word="banco"):
    seq=[]; meta=[]; word_ids=[]; word_set={word:0}
    for sense_label in ["A","B"]:
        tpls=POLYSEMY[word][f"templates_{sense_label}"]
        for _ in range(n_per_sense):
            sentence=fill_template(random.choice(tpls), sense_label, word)
            if word not in sentence.split():
                sentence = sentence + " y " + word
            toks=sentence.split()
            for t in toks:
                seq.append(t); meta.append(sense_label if t==word else "O"); word_ids.append(word_set.get(t,-1))
    return seq, list(dict.fromkeys(seq)), word_set, meta, word_ids

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
    def predict(self, omega_ctx):
        scores={w:dot(omega_ctx,self.emb[w]) for w in self.vocab}
        return max(scores, key=scores.get)

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

def run_baseline(seq, meta, word, W=8, epochs=5):
    vocab=sorted(set(seq))
    t=SimpleTransformer(vocab, D=16, lr=0.05)
    for ep in range(epochs):
        for i in range(W, len(seq)):
            ctx=seq[max(0,i-W):i]
            omega=t.forward(ctx)
            if seq[i] in t.emb:
                for d in range(16):
                    t.emb[seq[i]][d]+=t.lr*(omega[d]-t.emb[seq[i]][d])
    correct=0; total=0
    for i in range(len(seq)):
        if seq[i]==word:
            ctx=seq[max(0,i-W):i]
            omega=t.forward(ctx)
            ca=cos(omega,[0.5]*16); cb=cos(omega,[-0.5]*16)
            pred="A" if ca>=cb else "B"
            if pred==meta[i]: correct+=1
            total+=1
    return correct/total if total>0 else 0.0

def run_loop_no_decoder(seq, meta, word, W=8, epochs_outer=3, epochs_inner=5):
    vocab=sorted(set(seq))
    t=SimpleTransformer(vocab, D=16, lr=0.05)
    r=RootMemory(t, D=16, beta_anchor=0.2, beta_repulse=0.05, theta=0.8)
    seedA=[0.5]*16; seedB=[-0.5]*16
    r.seed_word(word,[seedA,seedB])
    # entrenamiento loop (sin decodificador)
    for ep_outer in range(epochs_outer):
        for ep_inner in range(epochs_inner):
            for i in range(W, len(seq)):
                w=seq[i]
                if w!=word: continue
                ctx=seq[max(0,i-W):i]
                omega=t.forward(ctx)
                r.update(word, ctx)
                # memoria: actualizar omega con foco (sin decodificador)
                A,B=r.sub[word]
                foco=r.foco[word]
                for dim in range(16):
                    omega[dim]+=foco*(A[dim]-B[dim])
                # volcar omega a transformer.omega para esa palabra
                t.omega[word]=list(omega)
        # re-entrenar transformer sobre secuencia
        for i in range(W, len(seq)):
            ctx=seq[max(0,i-W):i]
            omega=t.forward(ctx)
            if seq[i] in t.emb:
                for dim in range(16):
                    t.emb[seq[i]][dim]+=0.05*(omega[dim]-t.emb[seq[i]][dim])
    # evaluar
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
    print("=== v0.25 v8b LOOP SIN DECODIFICADOR ===")
    seq,vocab,word_set,meta,word_ids=build_corpus(n_per_sense=60, word="banco")
    word="banco"
    print(f"seq={len(seq)} vocab={len(vocab)}")
    baseline=run_baseline(seq,meta,word,W=8,epochs=5)
    loop=run_loop_no_decoder(seq,meta,word,W=8,epochs_outer=3,epochs_inner=5)
    print(f"baseline acc_decision={baseline:.3f}")
    print(f"loop acc_decision={loop:.3f}")
    mejora=loop-baseline
    print(f"mejora={mejora:.3f}")
    if loop>baseline+0.02:
        veredicto="LOOP MEJORA: sin decodificador, el ciclo suma señal."
    elif loop<baseline-0.02:
        veredicto="LOOP EMPEORA: el rompedor no era el decodificador."
    else:
        veredicto="LOOP INDISTINGUIBLE."
    print(veredicto)
    out=dict(experiment="v0.25_v8b_loop_sin_decoder", word=word,
             params=dict(W=8,epochs_outer=3,epochs_inner=5),
             results=dict(baseline=baseline, loop=loop, mejora=mejora, veredicto=veredicto))
    json.dump(out,open("results_v25_v8b.json","w"),indent=2)
    print("-> results_v25_v8b.json")
if __name__=="__main__":
    main()
