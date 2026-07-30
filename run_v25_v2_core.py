#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v2_core — Experimento canónico importando dscng_core."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dscng_core import SimpleTransformer, RootMemory, MetricLogger, build_polysemy_corpus

def main():
    word="banco"
    seq,meta,word,labels_map=build_polysemy_corpus(word=word, n_per_sense=1000)
    vocab=sorted(set(seq))
    t=SimpleTransformer(vocab, D=16, lr=0.05)
    r=RootMemory(D=16, lr=0.05, beta_anchor=0.2, beta_repulse=0.05, theta=0.8)
    ml=MetricLogger()
    hit=total=0
    for i in range(len(seq)):
        x=t.contexto(seq[:i+1])
        if seq[i]==word:
            A=x[:]; B=[-v for v in x]
            r.enraizar(A,B)
            y=labels_map[meta[i]]
            total+=1
            if (r.last_veredicto=='A' and y==1) or (r.last_veredicto=='B' and y==0):
                hit+=1
            acc=hit/total if total else 0.0
            ml.log(i, acc_pred=acc, acc_gt=acc, dolor=r.dolor, foco_acc=max(r.foco.values()), W_actual=r.W_actual)
    ml.to_json("results_v25_v2_core.json")
    print(json.dumps(ml.summary(), indent=2))

if __name__=="__main__":
    main()
