#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v22 — Coherencia de dominio generativa (robusta).
Tarea: generar texto coherente con un dominio (A/B) a partir de semillas.
Metrica: score de overlap semántico externo, promediado sobre múltiples semillas.
NO mide separación A/B interna; mide comportamiento coherente externo.
"""
import os, sys, random, json
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dscng_core import build_polysemy_corpus

WORD = "banco"

SEEDS = {
    "A": ["banco", "dinero", "ahorro", "cuenta", "plata", "interes", "tarjeta", "retiro"],
    "B": ["banco", "rio", "agua", "pez", "orilla", "puente", "corriente", "proa"],
}

def build_domain_vocab_examples(labeled, domain_key, topn=150):
    vocab = Counter()
    for sense_label, toks in labeled:
        if sense_label != domain_key:
            continue
        for w in toks[1:]:
            if w == WORD:
                continue
            vocab[w] += 1
    return set([w for w, _ in vocab.most_common(topn)])

def train_ngram_examples(labeled, domain_key, n=2):
    model = defaultdict(Counter)
    for sense_label, toks in labeled:
        if sense_label != domain_key:
            continue
        for i in range(len(toks) - n + 1):
            ctx = tuple(toks[i:i+n-1])
            nxt = toks[i+n-1]
            model[ctx][nxt] += 1
    return {ctx: dict(c) for ctx, c in model.items()}

def generate(model, seed, steps=25):
    seq = list(seed)
    ctx = (seq[-1],)
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
    return seq[len(seed):]

def coherencia_score(generated, vocab_domain, min_len=5):
    if len(generated) < min_len:
        return 0.0
    hits = sum(1 for w in generated if w in vocab_domain)
    return hits / len(generated)

def main():
    random.seed(42)
    n = 1000
    seq, meta, word, labels_map = build_polysemy_corpus(WORD, n_per_sense=n, augmentation=True)

    def sentences_by_sense():
        sens = []
        cur = []
        cur_sense = None
        for tok, m in zip(seq, meta):
            if m in ("A", "B"):
                if cur_sense is None:
                    cur_sense = m
                cur.append(tok)
            else:
                if cur and cur_sense:
                    sens.append((cur_sense, cur))
                cur = []
                cur_sense = None
        if cur and cur_sense:
            sens.append((cur_sense, cur))
        return sens

    labeled = sentences_by_sense()
    domain_vocab = {
        "A": build_domain_vocab_examples(labeled, "A", topn=150),
        "B": build_domain_vocab_examples(labeled, "B", topn=150),
    }
    models = {
        "A": train_ngram_examples(labeled, "A", n=2),
        "B": train_ngram_examples(labeled, "B", n=2),
    }

    out = {
        "experiment": "v0.25_v22_coherencia_dominio",
        "n": n,
        "seeds": SEEDS,
        "gen": {},
        "score": {},
        "baseline_random": {},
    }
    for dom in ("A", "B"):
        gens = []
        scores = []
        for seed_word in SEEDS[dom]:
            g = generate(models[dom], [seed_word], steps=25)
            s = coherencia_score(g, domain_vocab[dom])
            gens.append(" ".join(g))
            scores.append(s)
        out["gen"][dom] = gens
        out["score"][dom] = round(sum(scores) / len(scores), 4)
        rand_vocab = set(random.sample(list(domain_vocab[dom] | domain_vocab["A" if dom == "B" else "B"]), min(150, len(domain_vocab[dom])))) if domain_vocab[dom] else set()
        out["baseline_random"][dom] = round(coherencia_score(gens[0].split(), rand_vocab), 4) if rand_vocab else 0.0

    out["veredicto"] = (
        "FUNCIONAL" if min(out["score"].values()) >= 0.25 else
        "PARCIAL" if min(out["score"].values()) >= 0.12 else
        "NO FUNCIONAL"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open("results_v25_v22.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("-> results_v25_v22.json")

if __name__ == "__main__":
    main()
