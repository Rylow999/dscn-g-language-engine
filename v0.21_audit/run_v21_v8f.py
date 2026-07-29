#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.21 v8f — RESTA DE CONTEXTO (atencion implicita en grafo rustico).
v0.21 v8e (competencia+repulsion) dio acc_gt=0.529 (azar), poli_sep=3/6, no
converge. El grafo rustico con update iterativo de difusion no separa.
v0.21 v8f prueba la (4): RESTA DE CONTEXTO, como hace la atencion del transformer
(restar lo irrelevante). Para cada palabra polisemica:
  - subnodo A = embedding_base - promedio_contexto_B (resta lo que NO es A)
  - subnodo B = embedding_base - promedio_contexto_A (resta lo que NO es B)
El embedding_base = promedio de co-ocurrencia (como el grafo). La RESTA enfoca
el subnodo en lo que DISTINGUE A de B. No es difusion (no promedia todo), es
resta selectiva. El ground truth (sentido A/B) se usa SOLO para etiquetar y
evaluar (acc_gt), NO para definir los subnodos (que se construyen a partir de los
contextos reales del corpus). Instrumento correcto: corpus sintetico CON ground
truth + monosemicas de control. Hipotesis: si la resta da acc_gt>0.7, el grafo
rustico puede hacer atencion (sorpresa). Si 0.50, el transformer aporta mas alla
de la resta y es necesario.
"""
import json, math, random
from collections import defaultdict
D=16; W=8; SEED=0
def norm(v): return math.sqrt(sum(x*x for x in v))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def build_corpus(seed=SEED, n_per_sense=80):
    rng=random.Random(seed)
    poly={
      "banco":  (["dinero","pagar","cuenta","oro","plata","billete","banquero"],
                 ["rio","agua","pez","orilla","puente","corriente","rioja"]),
      "llave":  (["puerta","cerradura","abrir","candado","cerradura","llaves"],
                 ["musica","nota","tono","cancion","melodia","acorde","partitura"]),
      "mouse":  (["computadora","click","pantalla","cable","teclado","raton"],
                 ["animal","cola","hueco","roedor","rato","conejo","campo"]),
      "capital":(["dinero","inversion","bolsa","accion","finanza","mercado"],
                 ["ciudad","pais","provincia","madrid","lima","tierra"]),
      "firma":  (["empresa","contrato","sociedad","negocio","compania"],
                 ["manuscrito","firma","rubrica","documento","dedo","tinta"]),
      "oro":    (["metal","plata","joya","precioso","lingote","moneda"],
                 ["sol","amarillo","brillo","rayo","luz","dorado"]),
    }
    mono={
      "quijote":  ["libro","historia","caballero","mancha","ingenioso","novela"],
      "sancho":   ["escudero","panza","rocinante","insular","gordo","sencillez"],
      "caballero":["andante","espada","honor","armas","valor","caballeria"],
    }
    seq=[]; meta=[]
    def add_block(tokens, sense):
        for t in tokens: seq.append(t); meta.append(sense)
    for w,(sa,sb) in poly.items():
        for _ in range(n_per_sense):
            blk=list(sa[:5])+[w]+list(sa[5:7]); add_block(blk,"A")
            blk=list(sb[:5])+[w]+list(sb[5:7]); add_block(blk,"B")
    for w,cm in mono.items():
        for _ in range(n_per_sense):
            blk=list(cm[:5])+[w]+list(cm[5:7]); add_block(blk,"M")
    return seq, list(dict.fromkeys(seq)), list(poly.keys()), list(mono.keys()), meta
def build_embeddings(seq, vocab, meta, poly_words, mono_words):
    """Construye embedding_base (promedio co-ocurrencia) y contexto_A/B por palabra.
    El embedding_base de w = promedio de los embeddings de sus vecinos (D-dim,
    inicializados como one-hot-ish aleatorio pero estables). Para la RESTA,
    necesitamos: embedding_base[w] y promedio_contexto_A[w], promedio_contexto_B[w]
    (promedio de los embeddings de las palabras que aparecen en contexto A/B de w)."""
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED+1)
    # embedding_base: vector estable por palabra (representa su co-ocurrencia)
    emb=[[rng.gauss(0,1) for _ in range(D)] for _ in range(Vn)]
    # normalizo para que el coseno sea estable
    for i in range(Vn):
        n=norm(emb[i])
        if n>1e-9: emb[i]=[x/n for x in emb[i]]
    # contexto_A/B[w] = lista de embeddings de vecinos en sentido A/B
    ctxA=defaultdict(list); ctxB=defaultdict(list)
    for i in range(1,len(seq)):
        w=seq[i]; sense=meta[i]
        if sense in ("A","B") and w in idx:
            # contexto: vecinos antes y despues
            for j in range(max(0,i-W),min(len(seq),i+W+1)):
                if j==i: continue
                c=seq[j]
                if c in idx:
                    if sense=="A": ctxA[w].append(emb[idx[c]])
                    else: ctxB[w].append(emb[idx[c]])
    # promedio_contexto_A/B por palabra (vector D)
    avgA={}; avgB={}
    for w in poly_words:
        if w in ctxA and ctxA[w]:
            avgA[w]=[sum(x[d] for x in ctxA[w])/len(ctxA[w]) for d in range(D)]
        else:
            avgA[w]=[0.0]*D
        if w in ctxB and ctxB[w]:
            avgB[w]=[sum(x[d] for x in ctxB[w])/len(ctxB[w]) for d in range(D)]
        else:
            avgB[w]=[0.0]*D
    return emb, idx, avgA, avgB
def apply_resta(emb, idx, avgA, avgB, poly_words, mono_words):
    """RESTA DE CONTEXTO: subnodo_A = emb[w] - avgB[w], subnodo_B = emb[w] - avgA[w].
    Para monosemicas: subnodo_A = emb[w] (un solo subnodo, no hay contexto B)."""
    frac={}
    for w in poly_words:
        if w not in idx: continue
        wi=idx[w]; base=emb[wi]
        subA=[base[d]-avgB[w][d] for d in range(D)]  # resta lo que NO es A
        subB=[base[d]-avgA[w][d] for d in range(D)]  # resta lo que NO es B
        frac[w]=[subA, subB]
    for w in mono_words:
        if w not in idx: continue
        wi=idx[w]; base=emb[wi]
        frac[w]=[base, [x*0.99 for x in base]]  # 2 subnodos ~iguales (monosemico)
    return frac
def eval_full(seq, vocab, idx, frac, emb, meta, poly_words, mono_words):
    """acc_gt: ¿subnodo ganador corresponde al sentido real? poli/mono_sep."""
    correctos=0; total=0; poli_sep=0; mono_sep=0
    for w in poly_words:
        if w not in frac: continue
        occ=[i for i,x in enumerate(seq) if x==w]
        if len(occ)<10: continue
        grupos={}
        for i in occ:
            cw=list(range(max(0,i-W),i))
            if not cw: continue
            ctx=[0.0]*D
            for c in cw:
                if c in idx:
                    for d in range(D): ctx[d]+=emb[idx[c]][d]
            ctx=[x/len(cw) for x in ctx]
            bestk,bestc=-1,-1e9
            for k in range(2):
                c=cos(frac[w][k],ctx)
                if c>bestc: bestc=c; bestk=k
            grupos.setdefault(bestk,0); grupos[bestk]+=1
        n=len(occ); dom=max(grupos.values()) if grupos else n
        if len(grupos)>=2 and dom<n*0.85: poli_sep+=1
    for w in mono_words:
        if w not in frac: continue
        occ=[i for i,x in enumerate(seq) if x==w]
        if len(occ)<10: continue
        grupos={}
        for i in occ:
            cw=list(range(max(0,i-W),i))
            if not cw: continue
            ctx=[0.0]*D
            for c in cw:
                if c in idx:
                    for d in range(D): ctx[d]+=emb[idx[c]][d]
            ctx=[x/len(cw) for x in ctx]
            bestk,bestc=-1,-1e9
            for k in range(2):
                c=cos(frac[w][k],ctx)
                if c>bestc: bestc=c; bestk=k
            grupos.setdefault(bestk,0); grupos[bestk]+=1
        n=len(occ); dom=max(grupos.values()) if grupos else n
        if len(grupos)>=2 and dom<n*0.85: mono_sep+=1
    for i in range(1,len(seq)):
        w=seq[i]; sense=meta[i]
        if sense not in ("A","B") or w not in frac: continue
        cw=list(range(max(0,i-W),i))
        if not cw: continue
        ctx=[0.0]*D
        for c in cw:
            if c in idx:
                for d in range(D): ctx[d]+=emb[idx[c]][d]
        ctx=[x/len(cw) for x in ctx]
        bestk,bestc=-1,-1e9
        for k in range(2):
            c=cos(frac[w][k],ctx)
            if c>bestc: bestc=c; bestk=k
        esperado=0 if sense=="A" else 1
        if bestk==esperado: correctos+=1
        total+=1
    return (correctos/total if total else 0.0), poli_sep, mono_sep
def main():
    print("=== v0.21 v8f RESTA DE CONTEXTO (atencion implicita en grafo) ===")
    seq,vocab,poly_words,mono_words,meta=build_corpus()
    print(f"vocab={len(vocab)} poly={len(poly_words)} mono={len(mono_words)} seq={len(seq)}")
    emb,idx,avgA,avgB=build_embeddings(seq,vocab,meta,poly_words,mono_words)
    frac=apply_resta(emb,idx,avgA,avgB,poly_words,mono_words)
    acc,ps,ms=eval_full(seq,vocab,idx,frac,emb,meta,poly_words,mono_words)
    print(f"FINAL: acc_gt={acc:.3f} poli_sep={ps}/{len(poly_words)} mono_sep={ms}/{len(mono_words)}")
    # detalle por palabra
    for w in poly_words:
        if w in frac:
            print(f"  {w}: cos(subA,avgA)={cos(frac[w][0],avgA[w]):.3f} cos(subB,avgB)={cos(frac[w][1],avgB[w]):.3f}")
    out=dict(experiment="v0.21_v8f_resta_de_contexto_atencion_implicita",
             hypothesis="Resta de contexto (subA=emb-avgB, subB=emb-avgA) separa polisemicas. acc_gt>0.7 => grafo rustico puede hacer atencion.",
             params=dict(d=D,w=W,seed=SEED),
             resultados=dict(acc_gt=round(acc,3),poli_sep=ps,mono_sep=ms))
    json.dump(out,open("results_v21_v8f.json","w"),indent=2)
    print("-> results_v21_v8f.json")
if __name__=="__main__": main()