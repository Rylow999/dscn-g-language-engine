#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 v6 — ROOT con ATENCION SELECTIVA (no promedio).
v0.25 v5: el root con contexto PROMEDIO no detecta cambio (duda_detecta_cambio=
False) porque promediar W vecinos mezcla A/B. La causa: PROMEDIAR CONTEXTO
MEZCLA SENTIDOS.
v0.25 v6 prueba ATENCION SELECTIVA: en vez de promediar, el root asigna peso a
cada vecino segun su distintividad A/B (atencion). Un vecino distintivo de A
("dinero") pesa mas para A; filler ("el/la/de") pesa menos. Esto es lo que el
cerebro hace (atencion sobre features distintivos, no promedio ciego).
Mide: ¿acc_decision sube? ¿la duda detecta cambio de contexto?
"""
import json, math, random
from collections import defaultdict
D=16; W=8; SEED=0
def norm(v): return math.sqrt(sum(x*x for x in v))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def build_corpus(seed=SEED, n_per_sense=30):
    """Corpus de BLOQUES LARGOS: no intercala A/B, sino bloques de N
    ocurrencias de A seguidas y luego N de B. Así el contexto local es PURO
    (solo A o solo B), lo que permite medir si la atención selectiva detecta
    cambio de contexto."""
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
        for _ in range(n_per_sense):
            blk=list(sb[:5])+[w]+list(sb[5:7]); add_block(blk,"B")
    for w,cm in mono.items():
        for _ in range(n_per_sense):
            blk=list(cm[:5])+[w]+list(cm[5:7]); add_block(blk,"M")
    return seq, list(dict.fromkeys(seq)), list(poly.keys()), list(mono.keys()), meta, poly, mono
def build_embeddings_separados(poly_words, poly_dict, mono_words, mono_dict, vocab):
    """Embeddings que separan A/B PERFECTAMENTE (centros ortonormales)."""
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    centroA=[rng.gauss(0,1) for _ in range(D)]
    n=norm(centroA)
    if n>1e-9: centroA=[x/n for x in centroA]
    centroB=[rng.gauss(0,1) for _ in range(D)]
    dotAB=dot(centroA,centroB)
    centroB=[centroB[d]-dotAB*centroA[d] for d in range(D)]
    n=norm(centroB)
    if n>1e-9: centroB=[x/n for x in centroB]
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
    for t in ["el","la","de","y","en","con","por","un","una","que","los","las"]:
        emb[t]=[0.0]*D
    for w in vocab:
        if w not in emb: emb[w]=[rng.gauss(0,0.1) for _ in range(D)]
    emb_list=[emb[w] for w in vocab]
    return emb_list, idx, centroA, centroB
def root_atencion_selectiva(emb, idx, seq, meta, poly_words, centroA, centroB):
    """ROOT con ATENCION SELECTIVA (no promedio).
    Para cada ocurrencia de polisemia: en vez de promediar W vecinos, asigna
    peso a cada vecino segun distintividad A/B. Un vecino distintivo de A
    ("dinero") pesa mas para A; filler ("el") pesa menos.
    peso_j = exp(simA(v_j,centroA) - simB(v_j,centroB))  (distintividad)
    contexto_selectivo = suma(peso_j * v_j) / suma(peso_j)
    Esto es atencion: el root SELECCIONA features distintivas, no promedia."""
    dolor_en_cambio=0.0; cambio_count=0
    dolor_en_estable=0.0; estable_count=0
    acc_decision=0; decision_total=0
    prev_sentido=None
    for i in range(W,len(seq)):
        w=seq[i]; sense=meta[i]
        if w not in idx or sense not in ("A","B"): continue
        cw=list(range(max(0,i-W),i))
        if not cw: continue
        # ATENCION SELECTIVA: peso por distintividad A/B de cada vecino
        pesos=[]; vecs=[]
        for j in cw:
            if seq[j] not in idx: continue
            vj=emb[idx[seq[j]]]
            simA=cos(vj,centroA); simB=cos(vj,centroB)
            # distintividad: cuanto mas simA>simB, mas distintivo de A
            distintividad=simA-simB
            peso=math.exp(distintividad)  # atencion: enfatiza distintivos
            pesos.append(peso); vecs.append(vj)
        if not pesos: continue
        sp=sum(pesos)
        ctx=[0.0]*D
        for peso,vj in zip(pesos,vecs):
            for d in range(D): ctx[d]+=peso*vj[d]
        ctx=[x/sp for x in ctx]
        # decision: simA vs simB del contexto selectivo
        simA=cos(ctx,centroA); simB=cos(ctx,centroB)
        decision=0 if simA>=simB else 1
        esperado=0 if sense=="A" else 1
        if decision==esperado: acc_decision+=1
        decision_total+=1
        # DOLOR: 1 - |simA - simB|
        dolor=1.0-abs(simA-simB)
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
    print("=== v0.25 v6 ROOT con ATENCION SELECTIVA (no promedio) ===")
    seq,vocab,poly_words,mono_words,meta,poly_dict,mono_dict=build_corpus()
    print(f"vocab={len(vocab)} poly={len(poly_words)} mono={len(mono_words)} seq={len(seq)}")
    emb,idx,centroA,centroB=build_embeddings_separados(poly_words,poly_dict,mono_words,mono_dict,vocab)
    print(f"cos(A,B)={cos(centroA,centroB):.3f} (centros separan)")
    r=root_atencion_selectiva(emb,idx,seq,meta,poly_words,centroA,centroB)
    print(f"acc_decision (¿root decide A/B con atencion?): {r['acc_decision']:.3f} (azar=0.50)")
    print(f"dolor_en_cambio: {r['dolor_en_cambio']:.3f} ({r['cambio_count']} casos)")
    print(f"dolor_en_estable: {r['dolor_en_estable']:.3f} ({r['estable_count']} casos)")
    print(f"duda_detecta_cambio: {r['duda_detecta_cambio']}")
    if r['acc_decision']>0.7 and r['duda_detecta_cambio']:
        veredicto="ATENCION SELECTIVA FUNCIONA: root decide A/B + detecta cambio"
    elif r['acc_decision']>0.7:
        veredicto="ATENCION SELECTIVA SEPARA A/B: root decide A/B (pero no detecta cambio)"
    elif r['duda_detecta_cambio']:
        veredicto="ATENCION SELECTIVA DETECTA CAMBIO: duda funciona (pero no decide A/B)"
    else:
        veredicto="ATENCION SELECTIVA NO FUNCIONA: ni separa ni detecta cambio"
    print(f"VEREDICTO: {veredicto}")
    out=dict(experiment="v0.25_v6_root_atencion_selectiva",
             hypothesis="La atencion selectiva (peso por distintividad A/B) permite al root decidir A/B y detectar cambio de contexto.",
             params=dict(d=D,w=W,seed=SEED),
             resultados=dict(r,veredicto=veredicto))
    json.dump(out,open("results_v25_v6.json","w"),indent=2)
    print("-> results_v25_v6.json")
if __name__=="__main__": main()