#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v12_core — DECODIFICADOR GENERATIVO SOBRE EMBEDDINGS.
Importa SkipGram y utils del core; no modifica el core.
"""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dscng_core import build_polysemy_corpus, MetricLogger, SkipGram, cos

WORD = "banco"
D = 16; LR = 0.05; WINDOW = 5; EPOCHS = 20

def main():
    print("=== v0.25 v12_core — DECODIFICADOR GENERATIVO SOBRE EMBEDDINGS ===")
    seq, meta, word, labels_map = build_polysemy_corpus(WORD, n_per_sense=300, augmentation=True)
    vocab = sorted(set(seq))
    sg = SkipGram(vocab, D=D, lr=LR, window=WINDOW, neg_samples=5)
    sg.fit(seq, epochs=EPOCHS)
    # decoder: dado vector de contexto, predecir siguiente palabra por similitud coseno
    test_ctxs = []
    test_targets = []
    for i in range(len(seq) - 1):
        if seq[i] == WORD and meta[i] in ("A", "B"):
            ctx = seq[max(0, i - WINDOW):i]
            tgt = seq[i + 1]
            if tgt in sg.emb:
                test_ctxs.append(ctx)
                test_targets.append(tgt)
    top1 = 0; top5 = 0; n = 0
    for ctx, tgt in zip(test_ctxs, test_targets):
        vec = [0.0] * D
        valid = 0
        for w in ctx[-WINDOW:]:
            if w in sg.emb:
                for d in range(D): vec[d] += sg.emb[w][d]
                valid += 1
        if valid > 0: vec = [x / valid for x in vec]
        sims = [(cos(vec, sg.emb.get(v, [0]*D)), v) for v in vocab]
        sims.sort(reverse=True)
        top5_words = [v for _, v in sims[:5]]
        top1_word = sims[0][1] if sims else ""
        if top1_word == tgt: top1 += 1
        if tgt in top5_words: top5 += 1
        n += 1
    acc_top1 = top1 / max(1, n)
    acc_top5 = top5 / max(1, n)
    ml = MetricLogger()
    ml.log(step=0, acc_pred=acc_top1, acc_gt=acc_top5, dolor=0.0, foco_acc=0.0, W_actual=8)
    print(f"top1={acc_top1:.4f} top5={acc_top5:.4f} (n={n})")
    veredicto = "FUNCIONAL" if acc_top5 >= 0.3 else "NO FUNCIONAL"
    print(f"VEREDICTO: {veredicto} — decoder por similitud embeddings sobre corpus sintetico.")
    ml.to_json("results_v25_v12_core.json")
    print("-> results_v25_v12_core.json")

if __name__ == "__main__":
    main()
