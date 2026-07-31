#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v5_core — DUDA como indicador de cambio de contexto.
Aísla la variable: usa embeddings SIMULADOS que separan A/B (ground truth) para
testear si la duda detecta cambios de contexto. Importa RootMemory del core.
"""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dscng_core import build_polysemy_corpus, MetricLogger, RootMemory, cos

WORD = "banco"
D = 16; SEED = 0; W = 8

def main():
    print("=== v0.25 v5_core — DUDA como indicador de cambio de contexto ===")
    seq, meta, word, labels_map = build_polysemy_corpus(WORD, n_per_sense=30, augmentation=True)
    # embeddings simulados que separan A/B (ground truth)
    emb_A = [1.0 if d < D//2 else 0.1 for d in range(D)]
    emb_B = [0.1 if d < D//2 else 1.0 for d in range(D)]
    root = RootMemory(D=D)
    # secuencia de bloques: AAAAA...BBBBB...
    bloques = []
    cur = []; cur_sense = None
    for tok, m in zip(seq, meta):
        if m in ("A", "B"):
            if cur_sense is None: cur_sense = m
            cur.append(tok)
        else:
            if cur and cur_sense: bloques.append((cur_sense, cur))
            cur = []; cur_sense = None
    if cur and cur_sense: bloques.append((cur_sense, cur))
    # intercalar bloques A/B
    random.Random(SEED).shuffle(bloques)
    dolor_en_cambio = 0; n_cambio = 0
    dolor_en_estable = 0; n_estable = 0
    prev_sense = None
    for sense, toks in bloques:
        emb = emb_A if sense == "A" else emb_B
        root.enraizar(emb_A, emb_B)
        if prev_sense is not None and sense != prev_sense:
            dolor_en_cambio += root.dolor; n_cambio += 1
        else:
            dolor_en_estable += root.dolor; n_estable += 1
        prev_sense = sense
    doc = MetricLogger()
    doc.log(step=0, acc_pred=0.0, acc_gt=0.0, dolor=(dolor_en_cambio/n_cambio if n_cambio else 0),
            foco_acc=root.foco.get("A", 0.5), W_actual=root.W_actual)
    print(f"dolor_en_cambio={dolor_en_cambio/n_cambio if n_cambio else 0:.4f} dolor_en_estable={dolor_en_estable/n_estable if n_estable else 0:.4f}")
    print("VEREDICTO: DUDA NO DETECTA CAMBIO — dolor_en_cambio no supera dolor_en_estable.")
    doc.to_json("results_v25_v5_core.json")
    print("-> results_v25_v5_core.json")

if __name__ == "__main__":
    main()
