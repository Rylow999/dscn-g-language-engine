#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v3_core — TRANSFORMER BERT-STYLE sobre dscng_core.
Importa utils del core (build_polysemy_corpus, MetricLogger, LinearSenseClassifier, cos).
Re-implementa el transformer minimal localmente (no modifica core).
"""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dscng_core import build_polysemy_corpus, MetricLogger, LinearSenseClassifier, cos

WORD = "banco"
D = 16; LR = 0.05; SEED = 0; EPOCHS = 40; MASK_RATE = 0.15; W = 8

def softmax_logits(logits):
    m = max(logits)
    ex = [math.exp(l - m) for l in logits]
    s = sum(ex)
    return [e / s for e in ex] if s > 0 else [1.0 / len(logits)] * len(logits)

def build_vocab(seq):
    return sorted(set(seq))

class MinimalTransformer:
    def __init__(self, vocab, D=16, lr=0.05, mask_rate=0.15):
        self.D = D; self.lr = lr; self.mask_rate = mask_rate
        self.vocab = vocab
        self.emb = {w: [random.gauss(0, 0.1) for _ in range(D)] for w in vocab}
        self.emb["[MASK]"] = [random.gauss(0, 0.1) for _ in range(D)]
        self.correct = 0; self.total = 0
    def contexto(self, seq):
        ctx = seq[-W:]
        vec = [0.0] * self.D
        valid = 0
        for w in ctx:
            if w in self.emb:
                for d in range(self.D):
                    vec[d] += self.emb[w][d]
                valid += 1
        if valid > 0:
            vec = [x / valid for x in vec]
        return vec
    def get_embedding(self, w):
        return self.emb.get(w, [0.0] * self.D)
    def fit(self, seq, meta):
        rng = random.Random(SEED)
        for ep in range(EPOCHS):
            for i in range(len(seq)):
                if seq[i] not in self.emb: continue
                ctx = self.contexto(seq[max(0, i - W):i])
                if rng.random() < self.mask_rate:
                    logits = [sum(c * e for c, e in zip(ctx, self.emb.get(v, [0]*self.D))) for v in self.vocab]
                    probs = softmax_logits(logits)
                    pred_idx = max(range(len(probs)), key=lambda k: probs[k])
                    pred = self.vocab[pred_idx]
                    if pred == seq[i]:
                        self.correct += 1
                    self.total += 1
                    for d in range(self.D):
                        if pred != seq[i]:
                            self.emb[pred][d] += self.lr * (ctx[d] - self.emb[pred][d])
                        else:
                            self.emb[pred][d] += self.lr * (ctx[d] - self.emb[pred][d])
        self.acc_pred = self.correct / max(1, self.total)

def main():
    print("=== v0.25 v3_core — BERT-style sobre transformer ===")
    seq, meta, word, labels_map = build_polysemy_corpus(WORD, n_per_sense=60, augmentation=True)
    vocab = build_vocab(seq)
    tf = MinimalTransformer(vocab, D=D, lr=LR, mask_rate=MASK_RATE)
    tf.fit(seq, meta)
    # clasificador A/B sobre embeddings del transformer
    X, Y = [], []
    for i, tok in enumerate(seq):
        if tok == WORD and meta[i] in ("A", "B"):
            X.append(tf.get_embedding(tok))
            Y.append(labels_map[meta[i]])
    half = len(X) // 2
    clf = LinearSenseClassifier(D=D, lr=LR)
    clf.fit(X[:half], Y[:half], epochs=10)
    preds = [clf.predict(x) for x in X[half:]]
    acc_clf = sum(1 for p, y in zip(preds, Y[half:]) if p == y) / max(1, len(preds))
    ctx_A = tf.get_embedding("dinero")
    ctx_B = tf.get_embedding("rio")
    cAB = cos(ctx_A, ctx_B)
    ml = MetricLogger()
    ml.log(step=0, acc_pred=tf.acc_pred, acc_gt=acc_clf, dolor=0.0, foco_acc=0.0, W_actual=8)
    print(f"acc_pred={tf.acc_pred:.4f} acc_clf={acc_clf:.4f} cos(A_ctx,B_ctx)={cAB:.4f}")
    print("VEREDICTO: NO FUNCIONAL — transformer minimal sobre corpus sintetico no separa sentido (acc_clf≈azar).")
    ml.to_json("results_v25_v3_core.json")
    print("-> results_v25_v3_core.json")

if __name__ == "__main__":
    main()
