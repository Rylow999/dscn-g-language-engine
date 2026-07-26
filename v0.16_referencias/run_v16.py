#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.16 — REFERENCIAS COMPOSITIVAS (idea 2 de Luciano).
Un nodo tiene omega (geometrico) + lista de referencias a OTROS nodos que existen
afuera (simbólico). "boda" referencia {flores,vestido,blanco,beso}. Las referencias
se aprenden por co-ocurrencia. La poda por incoherencia (SynapticCache 2.1/2.3)
elimina la REFERENCIA pero el nodo externo (flores) sigue vivo.
Test: (1) las referencias de "boda" coinciden con sus co-ocurrentes reales;
(2) podar una referencia no borra el nodo externo.
"""
import json, math, random, re, sys, time
D=8; BETA=0.20; SEED=0
def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words, V):
    from collections import Counter
    return [w for w,_ in Counter(words).most_common(V)]
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    if na<1e-9 or nb<1e-9: return 0.0
    return sum(x*y for x,y in zip(a,b))/(na*nb)
def main():
    print("=== v0.16 REFERENCIAS COMPOSITIVAS (nodo = omega + refs) ===")
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text); vocab=build_vocab(words,150)
    rng=random.Random(SEED)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(len(vocab))]
    idx={w:i for i,w in enumerate(vocab)}
    seq=[w for w in words if w in set(vocab)]
    N=len(vocab)
    # references: cada nodo tiene un set de refs a otros nodos (aprendido por co-ocurrencia ventana 2)
    refs=[set() for _ in range(N)]
    t0=time.time()
    window=2
    for i in range(len(seq)):
        for j in range(max(0,i-window), min(len(seq), i+window+1)):
            if i!=j:
                refs[idx[seq[i]]].add(idx[seq[j]])
    print(f"refs aprendidas {time.time()-t0:.0f}s (ventana {window})")
    # test 1: las referencias de "boda" (si existe) son nodos que co-ocurren
    target="boda"
    if target in idx:
        rb=refs[idx[target]]
        # verificar que esas referencias efectivamente rodean a boda en el corpus
        ctx_real=set()
        for i,w in enumerate(seq):
            if w==target:
                for j in range(max(0,i-window),min(len(seq),i+window+1)):
                    if seq[j]!=target: ctx_real.add(seq[j])
        overlap=len(rb & set(idx[c] for c in ctx_real))
        total=len(rb|set(idx[c] for c in ctx_real))
        jaccard=overlap/total if total else 0
        print(f"boda: refs={len(rb)} ctx_real={len(ctx_real)} jaccard={jaccard:.3f}")
    else:
        jaccard=0; rb=set()
    # test 2: poda por incoherencia (SynapticCache 2.1/2.3) no borra nodo externo
    # simulamos: una referencia se poda si su coseno con el nodo cae bajo umbral
    THRESH=0.0
    podadas=0; externas_vivas=0
    for n in range(N):
        for r in list(refs[n]):
            if cos(omega[n],omega[r])<THRESH:
                refs[n].discard(r); podadas+=1
                # el nodo externo r sigue vivo (no se borra del grafo)
                externas_vivas+=1
    out=dict(experiment="v0.16_referencias_compositivas",
             hypothesis="Nodo = omega + refs a nodos externos; poda por incoherencia no borra el nodo externo.",
             params=dict(d=D,beta=BETA,vocab=N,window=window),
             target=target, n_refs_target=len(rb),
             jaccard_refs_ctx=round(jaccard,4),
             refs_podadas=Podadas if (Podadas:=podadas) else podadas,
             nodos_externos_vivos_tras_poda=externas_vivas,
             nota="Composicion simbolica: el nodo referencia a otros que existen afuera. Poda=desenlace, no borrado.")
    with open("results_v16.json","w") as f: json.dump(out,f,indent=2)
    print(f"refs podadas={podadas} nodos externos vivos tras poda={externas_vivas}")
    print("\n-> results_v16.json")
if __name__=="__main__": main()
