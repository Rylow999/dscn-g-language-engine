#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.21 v8 — ANCHOR/RESTART + REPULSION SIBLING (fix oversmoothing, v0.21 v7).
Diagnostico de Luciano: la regla omega[a]=(1-beta)omega[a]+beta*omega[b] es
difusion de grafo (power iteration de cadena de Markov) -> oversmoothing: converge
al autovector dominante, mata la separacion (componente alta frecuencia) sin importar
D ni epocas. La separacion es SIEMPRE transitoria (v6 ep11, v7 ep1) y muere.
Dos arreglos SIN backprop (pedido de Luciano):
  (1) ANCHOR/RESTART (Personalized PageRank / APPNP):
        omega[a] = alpha*omega0[a] + (1-alpha)*[(1-beta)omega[a] + beta*omega[b]]
      el ancla omega0[a] es inerosionable por el promediado -> rompe convergencia.
  (2) REPULSION SIBLING (beta negativo hacia el hermano del mismo lema):
        omega[a][k] -= beta_rep * omega[a][j]   (j = sibling de k)
Barrido de ALPHA para ver como modifica el resultado. Curva 15 epocas sobre las 3
polisemicas; buscamos SEPARACION ESTABLE (no transitoria).
"""
import json, math, random, time
from collections import Counter
D=16; W=4; EPOCHS=15; K=2; BETA=0.10; BETA_REP=0.20
ALPHAS=[0.05, 0.10, 0.20]; SEED=0
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
            ctx=[rng.choice(filler) for _ in range(3)] + list(sa[:3]) + [w] + list(sa[1:3]); seq+=ctx
        for _ in range(n_per_sense):
            ctx=[rng.choice(filler) for _ in range(3)] + list(sb[:3]) + [w] + list(sb[1:3]); seq+=ctx
    rng.shuffle(seq)
    vocab=list(dict.fromkeys(seq))
    return seq, vocab, list(poly.keys())
def train(seq,vocab,poly_words,alpha):
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    frac=[[[rng.gauss(0,1) for _ in range(D)] for _ in range(K)] for _ in range(Vn)]
    omega0=[[list(o) for o in frac[wi]] for wi in range(Vn)]  # ancla inicial
    curva=[]
    for ep in range(EPOCHS):
        for i in range(1,len(seq)):
            a=idx[seq[i-1]]; b=idx[seq[i]]
            tb=frac[b][0]
            for k in range(K):
                # (1-ALPHA) * difusion hacia target
                new=[(1-BETA)*frac[a][k][d]+BETA*tb[d] for d in range(D)]
                # (2) ANCHOR: alpha * omega0
                new=[alpha*omega0[a][k][d]+(1-alpha)*new[d] for d in range(D)]
                # (3) REPULSION SIBLING: beta_rep * hermano (beta negativo)
                j=1-k
                nj=norm(frac[a][j])
                if nj>1e-9:
                    new=[new[d]-BETA_REP*(frac[a][j][d]/nj) for d in range(D)]
                frac[a][k]=new
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
    return frac, curva
def main():
    print("=== v0.21 v8 ANCHOR/RESTART + REPULSION SIBLING (barrido alpha) ===")
    print(">> FIX OVERSMOOTHING (diagnostico Luciano): anchor rompe convergencia")
    print(">> al autovector dominante; repulsion sibling evita fusion de sentidos.")
    t0=time.time()
    seq,vocab,poly_words=build_contrast_corpus()
    print(f"corpus={len(seq)} tok, vocab={len(vocab)} (mide {poly_words})")
    resultados={}
    for alpha in ALPHAS:
        frac,curva=train(seq,vocab,poly_words,alpha)
        resultados[str(alpha)]={"curva":[{"epoca":e,"separadas":ok,"total":tot} for e,ok,tot in curva],
                                 "ultima":curva[-1][1], "estable":curva[-1][1]==len(poly_words)}
        print(f"  alpha={alpha}: ep1={curva[0][1]}/{curva[0][2]} ep15={curva[-1][1]}/{curva[-1][2]}")
    print(f"train total {time.time()-t0:.0f}s")
    out=dict(experiment="v0.21_v8_anchor_restart_repulsion_sibling",
             hypothesis="Anchor (alpha*omega0) + repulsion sibling (beta_rep) rompen el oversmoothing de la difusion: la separacion de sentidos se MANTIENE estable, no transitoria.",
             params=dict(d=D,window=W,epochs=EPOCHS,k=K,beta=BETA,beta_rep=BETA_REP,alphas=ALPHAS),
             por_alpha=resultados,
             veredicto=("SEPARACION ESTABLE (sin transformer)" if any(r["estable"] for r in resultados.values()) else "aun colapsa"))
    json.dump(out,open("results_v21_v8.json","w"),indent=2)
    print("\n-> results_v21_v8.json")
if __name__=="__main__": main()
