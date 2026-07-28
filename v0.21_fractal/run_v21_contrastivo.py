#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.21 v7 — CORPUS CONTRASTIVO + REPULSION EXPLICITA (fix de v6).
v0.21 v6 llego a 50/2403 separadas en ep11 pero recolapso a 0 (competencia suave
inestable) Y el vocab se inflo a 2403 (filler repetidas) -> el test media filler, no
las 3 polisemicas. Correcciones:
  - Vocab = SOLO las 3 polisemicas + un set FIJO de filler como contexto. El test de
    separacion MIDE SOLO las 3 palabras polisemicas (no las 2400 filler).
  - REPULSION EXPLICITA (codebook loss de VQ): cada subnodo de una palabra se aleja
    de los OTROS subnodos de la MISMA palabra, para que la separacion persista.
  - Curva de 15 epocas sobre las 3 polisemicas.
Hipotesis: con repulsion el grafo mantiene la separacion (no recolapsa) y el test
mide lo que corresponde. Si las 3 se separan y SE MANTIENEN -> el grafo rustico D=16
SÍ puede (tu intuicion), el limite era corpus/implementacion, no D=16.
"""
import json, math, random, time
from collections import Counter
D=16; W=4; EPOCHS=15; K=2; BETA=0.10; N_DEAD=50; REP=0.05
T_INIT=0.6; T_MIN=0.05; SEED=0
def norm(v): return math.sqrt(sum(x*x for x in v))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def build_contrast_corpus(seed=SEED, n_per_sense=50):
    rng=random.Random(seed)
    poly={
      "banco": (["dinero","pagar","cuenta","oro","plata"], ["rio","agua","pez","orilla","puente"]),
      "llave": (["puerta","cerradura","abrir","candado"], ["musica","nota","tono","cancion"]),
      "mouse": (["computadora","click","pantalla","cable"], ["animal","cola","raton","hueco"]),
    }
    filler=["el","la","de","y","en","con","por","un","una","que","los","las"]
    seq=[]
    for w,(sa,sb) in poly.items():
        for _ in range(n_per_sense):
            ctx=[rng.choice(filler) for _ in range(3)] + list(sa[:3]) + [w] + list(sa[1:3])
            seq+=ctx
        for _ in range(n_per_sense):
            ctx=[rng.choice(filler) for _ in range(3)] + list(sb[:3]) + [w] + list(sb[1:3])
            seq+=ctx
    rng.shuffle(seq)
    # vocab = TODAS las palabras unicas del corpus (el grafo aprende co-ocurrencia
    # de todo: polisemicas + palabras de sus sentidos + filler). El test de
    # separacion MIDE SOLO las polisemicas (poly_words), no el resto.
    vocab=list(dict.fromkeys(seq))  # preserva orden, sin repetir
    return seq, vocab, list(poly.keys())
def train(seq,vocab,poly_words):
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    frac=[[[rng.gauss(0,1) for _ in range(D)] for _ in range(K)] for _ in range(Vn)]
    dead=[[[0,-1e9,None] for _ in range(K)] for _ in range(Vn)]
    total_steps=EPOCHS*(len(seq)-1); si=0
    curva=[]
    for ep in range(EPOCHS):
        for i in range(1,len(seq)):
            a=idx[seq[i-1]]; b=idx[seq[i]]
            ctx_words=list(range(max(0,i-W),i))
            if ctx_words:
                ctx=[0.0]*D
                for c in ctx_words:
                    o=frac[idx[seq[c]]][0]
                    for d in range(D): ctx[d]+=o[d]
                ctx=[x/len(ctx_words) for x in ctx]
            else:
                ctx=[0.0]*D
            T=T_MIN+(T_INIT-T_MIN)*(1.0 - si/total_steps)
            sims=[cos(frac[a][k],ctx) for k in range(K)]
            mx=max(sims); ex=[math.exp((s-mx)/max(T,1e-6)) for s in sims]
            Z=sum(ex); w=[e/Z for e in ex]
            tb=frac[b][0]
            for k in range(K):
                frac[a][k]=[(1-BETA*w[k])*frac[a][k][d]+BETA*w[k]*tb[d] for d in range(D)]
                # REPULSION: subnodo_k se aleja de los OTROS subnodos de la MISMA palabra
                for j in range(K):
                    if j!=k:
                        na=norm(frac[a][k]); nj=norm(frac[a][j])
                        if na>1e-9 and nj>1e-9:
                            frac[a][k]=[frac[a][k][d]-REP*(frac[a][j][d]/nj) for d in range(D)]
            wk=max(range(K), key=lambda k:w[k])
            for k in range(K):
                if k==wk: dead[a][k][0]=i
                else:
                    c=cos(frac[a][k],ctx)
                    if c>dead[a][k][1]:
                        dead[a][k][1]=c; dead[a][k][2]=list(ctx)
            for k in range(K):
                if i-dead[a][k][0] > N_DEAD and dead[a][k][2] is not None:
                    frac[a][k]=[dead[a][k][2][d]+0.05*rng.gauss(0,1) for d in range(D)]
                    dead[a][k][0]=i; dead[a][k][1]=-1e9; dead[a][k][2]=None
            si+=1
        # medir SOLO las 3 polisemicas
        ok=0
        for w in poly_words:
            occ=[i for i,x in enumerate(seq) if x==w]
            grupos={}
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
                    c=cos(frac[idx[w]][k],ctx)
                    if c>bestc: bestc=c; bestk=k
                grupos.setdefault(bestk,0); grupos[bestk]+=1
            if len(grupos)>=2 and max(grupos.values())<len(occ)*0.85:
                ok+=1
        curva.append((ep+1,ok,len(poly_words)))
    return frac, idx, curva
def main():
    print("=== v0.21 v7 CORPUS CONTRASTIVO + REPULSION (fix v6) ===")
    t0=time.time()
    seq,vocab,poly_words=build_contrast_corpus()
    frac,idx,curva=train(seq,vocab,poly_words)
    print(f"train {time.time()-t0:.0f}s | corpus={len(seq)} tok, vocab={len(vocab)} (mide {poly_words})")
    print("curva (epoca, separadas, total_polisemicas):")
    for ep,ok,tot in curva:
        print(f"  ep{ep}: {ok}/{tot}")
    out=dict(experiment="v0.21_v7_contrastivo_repulsion",
             hypothesis="Con repulsion explicita (codebook loss) + test solo sobre polisemicas, el grafo mantiene la separacion y no recolapsa. El grafo D=16 SÍ puede (tu intuicion).",
             corpus="sintetico contrastivo (banco/llave/mouse, 2 sentidos x 50, intercalados)",
             vocab_size=len(vocab), mide=poly_words,
             curva=[{"epoca":e,"separadas":ok,"total":tot} for e,ok,tot in curva],
             ultima_epoca_separadas=curva[-1][1], total=curva[-1][2],
             veredicto=("GRAFO SEPARA Y MANTIENE" if curva[-1][1]==len(poly_words) else "aun inestable"))
    json.dump(out,open("results_v21_contrastivo.json","w"),indent=2)
    print("\n-> results_v21_contrastivo.json")
if __name__=="__main__": main()
