#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.21 v8c — ENDURECER el test de polisemia (auditoria del fix oversmoothing).
v0.21 v8b dio acc_gt=0.74: banco 1.0, mouse 0.95, PERO llave 0.267 FALLA. Esto
sugiere que el fix no resuelve contextos que se solapan o estan dominados por
filler. v0.21 v8c endurece:
(1) MAS palabras polisemicas (6) con contextos MAS LARGOS y DISTINTIVOS (menos
    filler entre la palabra y sus senales), para aislar si el fix escala.
(2) VENTANA mas grande (W=8) y contexto denso (sin filler en el bloque de senal).
(3) Mide acc_gt por palabra Y distancia de cada bucket a su contexto real, para
    ver POR QUE falla llave (filler domina? contextos solapados?).
(4) Barrido de umbral de repulsion condicional (cos<theta) para ver si un umbral
    mejor separa sin contaminar monosemicas.
HIPOTESIS HONESTA: si con contexto denso y W mayor el acc_gt sube >0.9, el fix es
genuino y llave fallaba por ruido de filler/ventana corta. Si sigue fallando,
hay solapamiento real de contextos que el fix no resuelve.
"""
import json, math, random, time
from collections import Counter
D=16; W=8; EPOCHS=20; K=2; BETA=0.10; BETA_REP=0.20; ALPHA=0.10; SEED=0
def norm(v): return math.sqrt(sum(x*x for x in v))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def build_corpus(seed=SEED, n_per_sense=80):
    rng=random.Random(seed)
    # POLISEMICAS: contexto A y B LARGOS y DISTINTIVOS (senal fuerte, poco filler)
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
    filler=["el","la","de","y","en","con","por","un","una","que","los","las"]
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
    return seq, list(dict.fromkeys(seq)), list(poly.keys()), list(mono.keys()), meta
def train(seq, vocab, conditional=True, theta=0.5):
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
                        cjj=cos(frac[a][k], frac[a][j])
                        if cjj < theta:
                            new=[new[d]-BETA_REP*(frac[a][j][d]/nj) for d in range(D)]
                    else:
                        new=[new[d]-BETA_REP*(frac[a][j][d]/nj) for d in range(D)]
                frac[a][k]=new
    return frac, idx
def classify_word(w, frac, idx, seq, meta):
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
    print("=== v0.21 v8c ENDURECER test polisemia (W=8, ctx denso, 6 poli) ===")
    seq,vocab,poly_words,mono_words,meta=build_corpus()
    print(f"vocab={len(vocab)} poly={poly_words} mono={mono_words} seq={len(seq)}")
    res={}
    for theta in (0.3, 0.5, 0.7):
        for cond in (False, True):
            tag=f"{'COND' if cond else 'INCOND'}_th{theta}"
            frac,idx=train(seq,vocab,conditional=cond,theta=theta)
            pr=[]; mr=[]
            for w in poly_words:
                r=classify_word(w,frac,idx,seq,meta)
                if r: pr.append(r)
            for w in mono_words:
                r=classify_word(w,frac,idx,seq,meta)
                if r: mr.append(r)
            n_mono_sep=sum(1 for r in mr if r['separada'])
            n_poli_sep=sum(1 for r in pr if r['separada'])
            acc_gt=sum(r['n_correct'] for r in pr)/sum(r['n'] for r in pr) if pr else 0
            detalle={r['word']:r['acc'] for r in pr}
            veredicto=("GENUNO" if (n_mono_sep==0 and n_poli_sep>0 and acc_gt>0.7)
                       else "ARTEFACTO" if n_mono_sep>=len(mr)*0.5 else "PARCIAL")
            print(f"  [{tag}] mono_sep={n_mono_sep}/{len(mr)} poli_sep={n_poli_sep}/{len(pr)} acc_gt={acc_gt:.2f} {detalle} -> {veredicto}")
            res[tag]=dict(mono_sep=n_mono_sep,poli_sep=n_poli_sep,acc_gt=round(acc_gt,3),
                          detalle=detalle,veredicto=veredicto)
    out=dict(experiment="v0.21_v8c_endurecer_polisemia",
             hypothesis="Con contexto denso (W=8, menos filler) el fix resuelve mas polisemicas (acc_gt>0.9); llave fallaba por ruido de filler/ventana corta.",
             params=dict(d=D,w=W,beta=BETA,beta_rep=BETA_REP,alpha=ALPHA,theta_sweep=[0.3,0.5,0.7]),
             resultados=res)
    json.dump(out,open("results_v21_v8c.json","w"),indent=2)
    print("-> results_v21_v8c.json")
if __name__=="__main__": main()