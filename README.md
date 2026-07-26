# DSCN-G Language Engine

Experiments proving DSCN-G (Dual-State Cognitive Geometry) as a cognitive substrate
for a language engine — not a next-token predictor, but a system with persistent
memory, emergent categorization, and internal pain that self-preserves.

## What this is

Luciano Nieto's hypothesis: intelligence is not "predict the next word" (like
current chatbots) but a substrate that (1) remembers without deletion, (2)
categorizes what it processes, and (3) feels a pain-signal that makes it correct
itself to survive. This repo runs those claims as empirical experiments.

## Confirmed (measured, not claimed)

| Result | Evidence |
|--------|----------|
| Massive persistent memory | v0.3 REAL: 100% mass retained via hibernation (no deletion) |
| Emergent categorization | v0.9b: 92.67% accuracy deducing noun/verb from usage |
| Internal pain / self-preservation | v0.9c: vitality G 0.0 → 1.0 under learned correction |
| Next-token learning | v0.6a: 10.11% on Don Quijote (vocab 150) |
| Live memory by relevance | v0.10: SynapticCache hybrid score keeps mass active |

## Limit (honest)

Contextual fluency (disambiguation like "banco"=bank/seat) requires a learned
attention layer (transformer) with backprop. The DSCN-G graph alone cannot
polysense (one node = one sense). We implemented manual backprop in pure Python
(no numpy/torch available on the device); it trains (loss drops) but needs more
care (learned output head, lower lr, more epochs) or PyTorch to converge to
top-1. See v0.14 / v0.14b / v0.14c / v0.14d.

## Structure

- `README.md` — full state table, limits, roadmap (Spanish).
- `RESUMEN_NOCHE.md` — nightly summary (Spanish).
- `EXPLICACION_CRIOLO.md` — plain-language description.
- `v0.1_concept_proof` ... `v0.14d_backprop` — each experiment: `run_vXX.py` + `results_vXX.json`.
- `gpt1_paper.pdf`, `PANDORA_Resumen.md` — references.

## Run

All experiments are pure Python 3 (no numpy/torch needed for the graph tests):
    python3 run_v03real.py
    python3 run_v09c.py
    ...
Corpus: Don Quijote (Project Gutenberg, public domain).

## Status

Substrate (memory + categorization + pain) PROVEN. Context layer PENDING
(requires backprop care / PyTorch). Architecture: graph (memory/pain) +
transformer (context) as complementary layers, not competitors.
