#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.5b — romper el loop de v0.5 con ventana de contexto (W(t) de Pandora,
version cavernicola). El omega de estado acumula historial y se PENALIZA repetir
las ultimas W palabras. Asi el grafo se ve obligado a moverse a otro nodo.
"""
import json, math, random, sys
ALPHA=5.0; D=8; WINDOW=3
CORPUS=["el","gato","come","pez","la","casa","es","roja","el perro","corre"]

def omega_of(word,rng):
    r=random.Random(hash(word)%100000)
    return [r.gauss(0,1) for _ in range(D)]
def norm(v): return math.sqrt(sum(x*x for x in v))
def affinity(q,w):
    d=math.sqrt(sum((a-b)**2 for a,b in zip(q,w)))
    return math.exp(-ALPHA*d)

def build_vocab():
    rng=random.Random(1)
    return {w:omega_of(w,rng) for w in CORPUS}

def generate(vocab, seed_word, steps=8, window=WINDOW):
    cur=vocab[seed_word][:]
    out=[seed_word]
    for _ in range(steps):
        recientes=set(out[-window:]) if window>0 else set()
        bestw,bests=None,-1
        for w,o in vocab.items():
            if w in recientes: continue          # penaliza repetir en ventana
            s=affinity(cur,o)
            if s>bests: bests=s; bestw=w
        if bestw is None:                         # si todo esta en ventana, libera
            for w,o in vocab.items():
                s=affinity(cur,o)
                if s>bests: bests=s; bestw=w
        out.append(bestw)
        # acumula historial: el estado avanza mezclandose con la palabra dicha
        cur=[0.6*cur[k]+0.4*vocab[bestw][k] for k in range(D)]
    return out

def main():
    print("=== v0.5b — ventana de contexto (romper loop) ===")
    vocab=build_vocab()
    fr=generate(vocab,"el",steps=8)
    print("sin ventana (v0.5): el casa el casa el casa el")
    print(f"con ventana W={WINDOW}: {' '.join(fr)}")
    # metrica: repeticiones adyacentes
    adj=sum(1 for i in range(1,len(fr)) if fr[i]==fr[i-1])
    out=dict(experiment="v0.5b_context_window",
             hypothesis="Ventana de contexto + penalizacion de repeticion rompe el loop.",
             params=dict(alpha=ALPHA,d=D,window=WINDOW,corpus_size=len(CORPUS)),
             seed="el", frase=" ".join(fr), repeticiones_adyacentes=adj)
    with open("results_v05b.json","w") as f: json.dump(out,f,indent=2)
    print(f"repeticiones adyacentes: {adj}  -> {'LOOP ROTO' if adj==0 else 'aun repite'}")
    print("\n-> results_v05b.json")

if __name__=="__main__": main()
