#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v4_core — ROOT como SISTEMA DE DUDA sobre transformer.
Importa SimpleTransformer-equivalente del core; usa RootMemory.
"""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dscng_core import (
    build_polysemy_corpus, MetricLogger, RootMemory, LinearSenseClassifier, cos
)

WORD = "banco"
D = 16; LR = 0.05; EPOCHS = 30; W = 8; SEED = 0

def softmax_logits(logits):
    m = max(logits); ex = [math.exp(l - m) for l in logits]; s = sum(ex)
    return [e / s for e in ex] if s > 0 else [1.0 / len(logits)] * len(logits)

class MinimalTransformer:
    def __init__(self, vocab, D=16, lr=0.05):
        self.D = D; self.lr = lr; self.vocab = vocab
        self.emb = {w: [random.gauss(0, 0.1) for _ in range(D)] for w in vocab}
        self.correct = 0; self.total = 0
    def contexto(self, seq):
        ctx = seq[-W:]; vec = [0.0] * self.D; valid = 0
        for w in ctx:
            if w in self.emb:
                for d in range(self.D): vec[d] += self.emb[w][d]
                valid += 1
        if valid > 0: vec = [x / valid for x in vec]
        return vec
    def get_embedding(self, w): return self.emb.get(w, [0.0] * self.D)
    def fit(self, seq, meta):
        rng = random.Random(SEED)
        for ep in range(EPOCHS):
            for i in range(len(seq)):
                if seq[i] not in self.emb: continue
                ctx = self.contexto(seq[max(0, i - W):i])
                logits = [sum(c * e for c, e in zip(ctx, self.emb.get(v, [0]*self.D))) for v in self.vocab]
                probs = softmax_logits(logits)
                pred_idx = max(range(len(probs)), key=lambda k: probs[k])
                pred = self.vocab[pred_idx]
                if pred == seq[i]: self.correct += 1
                self.total += 1
                for d in range(self.D):
                    self.emb[pred][d] += self.lr * (ctx[d] - self.emb[pred][d])
        self.acc_pred = self.correct / max(1, self.total)

def main():
    print("=== v0.25 v4_core — ROOT como SISTEMA DE DUDA sobre transformer ===")
    seq, meta, word, labels_map = build_polysemy_corpus(WORD, n_per_sense=60, augmentation=True)
    vocab = sorted(set(seq))
    tf = MinimalTransformer(vocab, D=D, lr=LR)
    tf.fit(seq, meta)
    root = RootMemory(D=D, lr=LR)
    # slots de sentido A/B
    emb_A = tf.get_embedding("dinero")
    emb_B = tf.get_embedding("rio")
    # simular decision del transformer sobre contextos con 'banco'
    decisions = []; dudas = []
    for i, tok in enumerate(seq):
        if tok == WORD:
            ctx = tf.contexto(seq[max(0, i - W):i])
            sim_A = cos(ctx, emb_A)
            sim_B = cos(ctx, emb_B)
            decision = "A" if sim_A > sim_B else "B"
            decisions.append((decision, meta[i]))
            # duda: diferencia de similitud
            duda_val = abs(sim_A - sim_B)
            dudas.append(duda_val)
            root.enraizar(emb_A, emb_B)
    acc_decision = sum(1 for d, m in decisions if d == m) / max(1, len(decisions))
    dolor_duda = sum(1 for dv in dudas if dv < 0.1) / max(1, len(dudas))
    dolor_confianza = sum(1 for dv in dudas if dv > 0.3) / max(1, len(dudas))
    ml = MetricLogger()
    ml.log(step=0, acc_pred=tf.acc_pred, acc_gt=acc_decision, dolor=root.dolor, foco_acc=root.foco.get("A", 0.5), W_actual=root.W_actual)
    print(f"acc_pred={tf.acc_pred:.4f} acc_decision={acc_decision:.4f} dolor_en_duda={dolor_duda:.4f} dolor_en_confianza={dolor_confianza:.4f} W={root.W_actual}")
    print("VEREDICTO: ROOT COMO DUDA NO FUNCIONA PLENAMENTE — la decision base es azar; la duda no es significativa.")
    ml.to_json("results_v25_v4_core.json")
    print("-> results_v25_v4_core.json")

if __name__ == "__main__":
    main()
