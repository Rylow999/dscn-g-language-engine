#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.21 v8b — RE-MEDICION CON INSTRUMENTO CORRECTO (auditoria del fix oversmoothing).
El v0.21 v8 original era CIRCULAR: repulsion INCONDICIONAL forzaba a los 2 sub-
embeddings de CUALQUIER palabra a divergir, y el criterio 'separada' (2 buckets
<85%) no contrastaba con ground truth. Control de Luciano (2026-07-28): 4/5
monosemicas del Quijote (quijote, sancho, caballero, dijo) tambien daban
'separadas' -> el 39/40 era artefacto, no senal de sentido.

Este v0.21 v8b CORRIGE el instrumento:
(1) CORPUS SINTETICO CON GROUND TRUTH: cada polisemica tiene sentido A (ctx_A)
    y sentido B (ctx_B) explícitos. Medimos DESAMBIGUACION CONTRA VERDAD:
    ¿el bucket asignado a cada ocurrencia coincide con su sentido real?
(2) PALABRAS MONOSEMICAS DE CONTROL insertadas en contexto FIJO: deben quedar en
    1 bucket dominante (>85%). Si se reparten, la repulsion sigue contaminando.
(3) REPULSION CONDICIONAL (nuevo): en lugar de restar ciegamente el hermano en
    cada paso, solo aplicamos repulsion si los 2 buckets ya recibieron contextos
    DISTINTOS (senal de que hay 2 sentidos). Si un lema es monosémico, sus 2
    buckets reciben contexto igual => no hay repulsion => quedan fusionados (1
    bucket). Esto distingue mecanicamente polisemico de monosémico.

HIPOTESIS HONESTA: si la repulsion CONDICIONAL funciona, monosemicas quedan en 1
bucket y polisemicas se separan CON el bucket correcto. Si no, el fix de
oversmoothing por repulsion no sirve y hay que cambiar de estrategia.
"""
import json, math, random, time
from collections import Counter
D=16; W=4; EPOCHS=15; K=2; BETA=0.10; BETA_REP=0.20; ALPHA=0.10; SEED=0
def norm(v): return math.sqrt(sum(x*x for x in v))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def build_corpus(seed=SEED, n_per_sense=60):
    rng=random.Random(seed)
    poly={
      "banco": (["dinero","pagar","cuenta","oro","plata"], ["rio","agua","pez","orilla","puente"]),
      "llave": (["puerta","cerradura","abrir","candado"], ["musica","nota","tono","cancion"]),
      "mouse": (["computadora","click","pantalla","cable"], ["animal","cola","raton","hueco"]),
    }
    mono={
      "quijote": ["libro","historia","caballero","mancha","ingenioso"],
      "sancho": ["escudero","panza","rocinante","insular","gordo"],
      "caballero": ["andante","espada","honor","armas","valor"],
    }
    filler=["el","la","de","y","en","con","por","un","una","que","los","las"]
    seq=[]; meta=[]  # meta[i] = sentido real ('A'/'B'/'M') o None para filler
    def add_block(tokens, sense):
        for t in tokens:
            seq.append(t); meta.append(sense)
    for w,(sa,sb) in poly.items():
        for _ in range(n_per_sense):
            blk=[rng.choice(filler) for _ in range(3)]+list(sa[:3])+[w]+list(sa[1:3])
            add_block(blk,"A")
        for _ in range(n_per_sense):
            blk=[rng.choice(filler) for _ in range(3)]+list(sb[:3])+[w]+list(sb[1:3])
            add_block(blk,"B")
    for w,cm in mono.items():
        for _ in range(n_per_sense):
            blk=[rng.choice(filler) for _ in range(3)]+list(cm[:3])+[w]+list(cm[1:3])
            add_block(blk,"M")
    return seq, list(dict.fromkeys(seq)), list(poly.keys()), list(mono.keys()), meta
def train(seq, vocab, conditional=True):
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    frac=[[[rng.gauss(0,1) for _ in range(D)] for _ in range(K)] for _ in range(Vn)]
    omega0=[[list(o) for o in frac[wi]] for wi in range(Vn)]
    for ep in range(EPOCHS):
        for i in range(1,len(seq)):
            a=idx[seq[i-1]]; b=idx[seq[i]]; tb=frac[b][0]
            for k in range(K):
                new=[(1-BETA)*frac[a][k][d]+BETA*tb[d] for d in range(D)]
                new=[ALPHA*omega0[a][k][d]+(1-ALPHA)*new[d] for d in range(D)]
                j=1-k; nj=norm(frac[a][j])
                if nj>1e-9:
                    if conditional:
                        # REPULSION CONDICIONAL: solo si los 2 buckets ya tienen
                        # contexto DIVERGENTE (cos<0.5) => senal de 2 sentidos.
                        # Monosemico: ambos buckets reciben ctx igual => cos~1 => no repulsion.
                        cjj=cos(frac[a][k], frac[a][j])
                        if cjj < 0.5:
                            new=[new[d]-BETA_REP*(frac[a][j][d]/nj) for d in range(D)]
                    else:
                        new=[new[d]-BETA_REP*(frac[a][j][d]/nj) for d in range(D)]  # incondicional (v0.21 v8)
                frac[a][k]=new
    return frac, idx
def classify_word(w, frac, idx, seq, meta):
    """Para cada ocurrencia de w, asigna bucket por contexto y compara con meta.
    Devuelve: n, n_correct (bucket coincide con sentido A/B), dom_pct (bucket
    dominante), buckets dict, separada (<85% dom)."""
    if w not in idx: return None
    wi=idx[w]; occ=[i for i,x in enumerate(seq) if x==w]
    if len(occ)<10: return None
    grupos={}; correctos=0
    for i in occ:
        cw=list(range(max(0,i-W),i))
        if not cw: continue
        ctx=[0.0]*D
        for c in cw:
            o=frac[idx[seq[c]]][0]
            for d in range(D): ctx[d]+=o[d]
        ctx=[x/len(cw) for x in ctx]
        bestk,bestc=-1,-1e9
        for k in range(K):
            c=cos(frac[wi][k],ctx)
            if c>bestc: bestc=c; bestk=k
        grupos.setdefault(bestk,0); grupos[bestk]+=1
        # ground truth: sentido A->bucket esperado 0, B->1, M->cualquiera dominante
        sense=meta[i]
        if sense in ("A","B"):
            esperado=0 if sense=="A" else 1
            if bestk==esperado: correctos+=1
    n=len(occ)
    dom=max(grupos.values()) if grupos else n
    separada=(len(grupos)>=2 and dom<n*0.85)
    return dict(word=w,n=n,dom_pct=round(dom/n,3),buckets=dict(grupos),
                separada=separada,n_correct=correctos,acc=round(correctos/n,3))
def main():
    print("=== v0.21 v8b RE-MEDICION con instrumento correcto (conditional repulsion) ===")
    seq,vocab,poly_words,mono_words,meta=build_corpus()
    print(f"vocab={len(vocab)} poly={poly_words} mono={mono_words} seq={len(seq)}")
    res={}
    for cond in (False, True):
        tag="INCONDICIONAL" if not cond else "CONDICIONAL"
        frac,idx=train(seq,vocab,conditional=cond)
        pr=[]; mr=[]
        for w in poly_words:
            r=classify_word(w,frac,idx,seq,meta)
            if r: pr.append(r); print(f"  [{tag}] POLI {w:8} n={r['n']:3} dom={r['dom_pct']:.2f} sep={r['separada']} acc_gt={r['acc']}")
        for w in mono_words:
            r=classify_word(w,frac,idx,seq,meta)
            if r: mr.append(r); print(f"  [{tag}] MONO {w:8} n={r['n']:3} dom={r['dom_pct']:.2f} sep={r['separada']} (deberia ser sep=False)")
        n_mono_sep=sum(1 for r in mr if r['separada'])
        n_poli_sep=sum(1 for r in pr if r['separada'])
        acc_gt=sum(r['n_correct'] for r in pr)/sum(r['n'] for r in pr) if pr else 0
        veredicto=("GENUNO" if (n_mono_sep==0 and n_poli_sep>0 and acc_gt>0.7)
                   else "ARTEFACTO" if n_mono_sep>=len(mr)*0.5
                   else "PARCIAL")
        print(f"  [{tag}] mono_sep={n_mono_sep}/{len(mr)} poli_sep={n_poli_sep}/{len(pr)} acc_gt={acc_gt:.2f} -> {veredicto}")
        res[tag]=dict(mono=mr,poli=pr,n_mono_sep=n_mono_sep,n_poli_sep=n_poli_sep,
                      acc_gt=round(acc_gt,3),veredicto=veredicto)
    out=dict(experiment="v0.21_v8b_revision_fix_oversmoothing",
             hypothesis="Repulsion CONDICIONAL (solo si hay contexto diverso) separa polisemicas SIN contaminar monosemicas; el v0.21 v8 original era artefacto.",
             params=dict(d=D,beta=BETA,beta_rep=BETA_REP,alpha=ALPHA,conditional_vs_inconditional=True),
             resultados=res)
    json.dump(out,open("results_v21_v8b.json","w"),indent=2)
    print("-> results_v21_v8b.json")
if __name__=="__main__": main()
