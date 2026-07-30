#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v5 — DUDA como indicador de cambio de contexto (validacion B).
v0.25 v4: el root como sistema de duda NO funciona porque el transformer minimo
no separa sentido (acc_decision=0.544=azar, acc_pred=0.901 solo predice tokens).
La duda no es significativa si la decision base es azar.
v0.25 v5 valida la idea de Luciano (duda como indicador de cambio de contexto)
ISOLANDO la variable: usa embeddings que SEParen A/B (simulados con ground
truth, para aislar si la duda funciona sobre una representacion que separa).
Si la duda detecta cambios de contexto (dolor alto cuando cambia A↔B), la idea
funciona y el root como sistema de duda es viable (solo necesita transformer que
separe). Si no, la duda no es un buen indicador.
"""
import json, math, random
from collections import defaultdict
D=16; W=8; SEED=0
def norm(v): return math.sqrt(sum(x*x for x in v))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def build_corpus(seed=SEED, n_per_sense=60):
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
      "oro":    (["metal","plata","joya","precioso","lingote","moneda"],
                 ["sol","amarillo","brillo","rayo","luz","dorado"]),
    }
    mono={
      "quijote":  ["libro","historia","caballero","mancha","ingenioso","novela"],
      "sancho":   ["escudero","panza","rocinante","insular","gordo","sencillez"],
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
    return seq, list(dict.fromkeys(seq)), list(poly.keys()), list(mono.keys()), meta, poly, mono
def build_embeddings_separados(poly_words, poly_dict, mono_words, mono_dict, vocab):
    """Simula embeddings que SEParen A/B (usando ground truth para aislar).
    Cada palabra de sentido A tiene embedding cercano al centro_A (vector D),
    B cercano al centro_B. Los centros A/B son ortonormales (separan bien).
    Esto ISOLA la pregunta: ¿la duda del root detecta cambios de contexto sobre
    una representacion que separa? (no depende del transformer minimo)."""
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    # centros A y B ortonormales (separan maximamente)
    centroA=[rng.gauss(0,1) for _ in range(D)]
    n=norm(centroA)
    if n>1e-9: centroA=[x/n for x in centroA]
    centroB=[rng.gauss(0,1) for _ in range(D)]
    # hacer B ortonormal a A
    dotAB=dot(centroA,centroB)
    centroB=[centroB[d]-dotAB*centroA[d] for d in range(D)]
    n=norm(centroB)
    if n>1e-9: centroB=[x/n for x in centroB]
    # embeddings: palabras de A cerca de centroA, de B cerca de centroB
    emb={}
    for w in poly_words:
        sa,sb=poly_dict[w]
        for t in sa: emb[t]=[centroA[d]+rng.gauss(0,0.1) for d in range(D)]
        for t in sb: emb[t]=[centroB[d]+rng.gauss(0,0.1) for d in range(D)]
    for w in mono_words:
        cm=mono_dict[w]
        c=[rng.gauss(0,1) for _ in range(D)]
        n=norm(c)
        if n>1e-9: c=[x/n for x in c]
        for t in cm: emb[t]=[c[d]+rng.gauss(0,0.1) for d in range(D)]
    # palabras no distintivas (filler) en el origen
    for t in ["el","la","de","y","en","con","por","un","una","que","los","las"]:
        emb[t]=[0.0]*D
    # asegurar que todas las palabras del vocab tengan embedding
    for w in vocab:
        if w not in emb: emb[w]=[rng.gauss(0,0.1) for _ in range(D)]
    # convertir a lista indexada por idx (para root_duda_contexto)
    emb_list=[emb[w] for w in vocab]
    return emb_list, idx, centroA, centroB
def root_duda_contexto(emb, idx, seq, meta, poly_words, centroA, centroB):
    """ROOT como SISTEMA DE DUDA sobre embeddings que separan A/B.
    Para cada ocurrencia de polisemia: el contexto actual (promedio de embeddings
    de vecinos) se compara con centroA/centroB. Si el contexto es ambiguo (similar
    a ambos centros), duda alta → dolor. Si es claro (similar a un solo centro),
    confianza alta → dolor bajo. Mide:
    - dolor_en_cambio: ¿el dolor sube cuando el contexto cambia A↔B?
    - dolor_en_estable: ¿el dolor es bajo cuando el contexto es estable?
    - acc_decision: ¿el root decide A/B correctamente?"""
    dolor_en_cambio=0.0; cambio_count=0
    dolor_en_estable=0.0; estable_count=0
    acc_decision=0; decision_total=0
    prev_sentido=None
    for i in range(W,len(seq)):
        w=seq[i]; sense=meta[i]
        if w not in idx or sense not in ("A","B"): continue
        # CONTEXTO actual
        ctx=[0.0]*D
        for j in range(max(0,i-W),i):
            if seq[j] in idx:
                for d in range(D): ctx[d]+=emb[idx[seq[j]]][d]
        ctx=[x/max(1,len(range(max(0,i-W),i))) for x in ctx]
        # similitud al centro A y B
        simA=cos(ctx,centroA); simB=cos(ctx,centroB)
        # decision: A si simA>simB
        decision=0 if simA>=simB else 1
        esperado=0 if sense=="A" else 1
        if decision==esperado: acc_decision+=1
        decision_total+=1
        # DOLOR: 1 - |simA - simB| (si ambos son similares, alta duda)
        dolor=1.0-abs(simA-simB)
        # clasificar: cambio de contexto (A↔B) vs estable (mismo sentido)
        if prev_sentido is not None and sense!=prev_sentido:
            dolor_en_cambio+=dolor; cambio_count+=1
        elif prev_sentido is not None and sense==prev_sentido:
            dolor_en_estable+=dolor; estable_count+=1
        prev_sentido=sense
    acc_dec=acc_decision/decision_total if decision_total else 0.0
    dolor_cambio=dolor_en_cambio/cambio_count if cambio_count else 0.0
    dolor_estable=dolor_en_estable/estable_count if estable_count else 0.0
    return dict(acc_decision=acc_dec, dolor_en_cambio=dolor_cambio,
                dolor_en_estable=dolor_estable, cambio_count=cambio_count,
                estable_count=estable_count,
                duda_detecta_cambio=(dolor_cambio>dolor_estable+0.05))
def main():
    print("=== v0.25 v5 DUDA como indicador de cambio de contexto (representacion que separa) ===")
    seq,vocab,poly_words,mono_words,meta,poly_dict,mono_dict=build_corpus()
    print(f"vocab={len(vocab)} poly={len(poly_words)} mono={len(mono_words)} seq={len(seq)}")
    emb,idx,centroA,centroB=build_embeddings_separados(poly_words,poly_dict,mono_words,mono_dict,vocab)
    print(f"centros ortonormales: cos(A,B)={cos(centroA,centroB):.3f} (debe ser ~0)")
    r=root_duda_contexto(emb,idx,seq,meta,poly_words,centroA,centroB)
    print(f"acc_decision (¿root decide A/B?): {r['acc_decision']:.3f} (azar=0.50)")
    print(f"dolor_en_cambio (¿subte cuando cambia A↔B?): {r['dolor_en_cambio']:.3f} ({r['cambio_count']} casos)")
    print(f"dolor_en_estable (¿bajo cuando contexto estable?): {r['dolor_en_estable']:.3f} ({r['estable_count']} casos)")
    print(f"duda_detecta_cambio: {r['duda_detecta_cambio']}")
    if r['duda_detecta_cambio']:
        veredicto="DUDA DETECTA CAMBIO: el root distingue cambio de contexto (dolor sube A↔B)"
    else:
        veredicto="DUDA NO DETECTA CAMBIO: el dolor no distingue cambio de contexto"
    print(f"VEREDICTO: {veredicto}")
    out=dict(experiment="v0.25_v5_duda_indicador_cambio_contexto",
             hypothesis="La duda del root detecta cambios de contexto A↔B sobre una representacion que separa. Si dolor_en_cambio>dolor_en_estable, la duda es un buen indicador.",
             params=dict(d=D,w=W,seed=SEED),
             resultados=r)
    out["resultados"]["veredicto"]=veredicto
    json.dump(out,open("results_v25_v5.json","w"),indent=2)
    print("-> results_v25_v5.json")
if __name__=="__main__": main()