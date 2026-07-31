#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v22b — COHERENCIA DE DOMINIO sobre CORPUS REAL (Don Quijote).
Valida v22 sobre texto real: extrae contextos de 'banco' en DQ, k-means para
hallar clusters, y mide coherencia de generación por dominio vs vocabulario
real del corpus. NO usa acc_gt interna.
"""
import sys, os, math, random, json, re
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dscng_core import cos

WORD = "tiempo"
K = 2; W = 10; TOPN = 80; D = 16

def load_dq():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "donquijote.txt")
    if not os.path.exists(path):
        print("donquijote.txt NO disponible en home; aborta."); return None
    with open(path, encoding="utf-8") as f:
        text = f.read().lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    return text.split()

def build_vocab(seq, min_count=3):
    c = Counter(seq)
    return {w: i for i, w in enumerate(sorted([w for w, n in c.items() if n >= min_count]))}

def context_vector(ctx, vocab, D=16):
    vec = [0.0] * D
    for w in ctx:
        if w in vocab:
            idx = vocab[w] % D
            vec[idx] += 1.0
    return vec

def kmeans(X, k, epochs=30, seed=0):
    rng = random.Random(seed)
    if len(X) < k:
        return [0] * len(X), X[:k]
    centroids = [X[rng.randrange(len(X))] for _ in range(k)]
    labels = []
    for _ in range(epochs):
        labels = []
        for x in X:
            sims = [cos(x, c) for c in centroids]
            labels.append(sims.index(max(sims)))
        for ki in range(k):
            pts = [X[i] for i, l in enumerate(labels) if l == ki]
            if pts:
                centroids[ki] = [sum(p[d] for p in pts) / len(pts) for d in range(D)]
    return labels, centroids

def train_ngram_by_cluster(seq, positions, labels, k=2, n=2):
    models = {}
    cluster_tokens = {ki: [] for ki in range(k)}
    cluster_texts = {ki: [] for ki in range(k)}
    for pos, lbl in zip(positions, labels):
        ctx = seq[max(0, pos - W):pos]
        cluster_tokens[lbl].append(seq[pos])
        cluster_texts[lbl].append(" ".join(ctx[-3:]))
    for lbl, tokens in cluster_tokens.items():
        t = defaultdict(Counter)
        for w, wn in zip(tokens, tokens[1:]):
            t[w][wn] += 1
        models[lbl] = {w: {k2: v / sum(c.values()) for k2, v in c.items()} for w, c in t.items()}
    return models, cluster_tokens

def generate(model, seed_word, steps=25):
    seq = [seed_word]
    ctx = (seed_word,)
    for _ in range(steps):
        opts = model.get(ctx, {})
        if not opts:
            break
        total = sum(opts.values())
        r = random.random() * total
        acc = 0
        nxt = random.choice(list(opts.keys()))
        for w, c in opts.items():
            acc += c
            if r <= acc:
                nxt = w
                break
        seq.append(nxt)
        ctx = (nxt,)
    return seq[1:]

def build_domain_vocab(seq, positions, labels, k, topn=120):
    vocabs = {}
    for ki in range(k):
        tokens = []
        for pos, lbl in zip(positions, labels):
            if lbl != ki:
                continue
            ctx = seq[max(0, pos - W):pos]
            tokens.extend([w for w in ctx if w != WORD])
        c = Counter(tokens)
        vocabs[ki] = set([w for w, _ in c.most_common(topn)])
    return vocabs

def main():
    print("=== v0.25 v22b — Coherencia sobre Don Quijote REAL ===")
    seq = load_dq()
    if seq is None:
        return
    print(f" tokens DQ={len(seq)}")
    vocab = build_vocab(seq, min_count=3)
    print(f" vocab={len(vocab)}")
    X = []; positions = []
    for i in range(len(seq)):
        if seq[i] == WORD:
            ctx = seq[max(0, i - W):i]
            vec = context_vector(ctx, vocab, D)
            if sum(vec) > 0:
                X.append(vec); positions.append(i)
    print(f" contextos 'banco'={len(X)}")
    if len(X) < 10:
        print(" pocos contextos, aborta."); return
    labels, centroids = kmeans(X, k=K, epochs=30)
    counts = Counter(labels)
    print(f" clusters k={K}: {counts}")
    if min(counts.values()) < 5:
        print(" cluster demasiado chico, aborta."); return
    models, cluster_tokens = train_ngram_by_cluster(seq, positions, labels, k=K)
    domain_vocab = build_domain_vocab(seq, positions, labels, K, topn=TOPN)
    # generar por cluster y medir coherencia
    scores = {}
    gens = {}
    for ki in range(K):
        sample = random.choice(cluster_tokens[ki]) if cluster_tokens[ki] else WORD
        gen = generate(models[ki], sample, steps=25)
        gens[ki] = " ".join(gen)
        hits = sum(1 for w in gen if w in domain_vocab[ki])
        scores[ki] = hits / len(gen) if gen else 0
    # baseline random: overlap con vocab del otro cluster
    rand_scores = {}
    for ki in range(K):
        other = 1 - ki
        hits = sum(1 for w in gens[ki].split() if w in domain_vocab[other])
        rand_scores[ki] = hits / len(gens[ki].split()) if gens[ki].split() else 0
    print(f" scores={scores}")
    print(f" random_overlap={rand_scores}")
    veredicto = "FUNCIONAL" if min(scores.values()) >= 0.15 else "PARCIAL" if min(scores.values()) >= 0.08 else "NO FUNCIONAL"
    print(f" VEREDICTO: {veredicto}")
    out = {
        "experiment": "v0.25_v22b_coherencia_DQ",
        "n_clusters": K,
        "sizes": dict(counts),
        "scores": {str(k): round(v, 4) for k, v in scores.items()},
        "random_overlap": {str(k): round(v, 4) for k, v in rand_scores.items()},
        "gen": gens,
        "veredicto": veredicto,
    }
    with open("results_v25_v22b.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("-> results_v25_v22b.json")

if __name__ == "__main__":
    main()
