#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.22 v5 — CONTEXTOS MIXTOS + proyeccion SUAVE + MARGIN adaptativo.
v0.22 v4: proyeccion Hebb demasiado fuerte -> diffs siempre >0 -> duda=0.
Hipotesis: con proyeccion SUAVE (menos epochs Hebb) y contextos MIXTOS (ambos
sentidos presentes, ej 'banco del rio sacar dinero'), la duda DEBE emerger en lo
mixto y NO en lo univoco. Mide: duda en mixto vs univoco (debe ser mayor en mixto).
"""
import json, math, random, re, time
from collections import Counter
D=16; WIN=4; K=2; BETA=0.10; BETA_REP=0.20; ALPHA=0.10; LR_W=0.005; EPOCHS_W=1
PCT=80; SEED=0
def norm(v): return math.sqrt(sum(x*x for x in v)) or 1e-9
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def mat_vec(M,v): return [sum(M[i][j]*v[j] for j in range(D)) for i in range(D)]
def build_corpus(seed=SEED, n=40):
    rng=random.Random(seed)
    poly={"banco":(["dinero","pagar","cuenta","oro"],["rio","agua","pez","orilla"]),
          "llave":(["puerta","cerradura","abrir","candado"],["musica","nota","tono","cancion"]),
          "mouse":(["computadora","click","pantalla","cable"],["animal","cola","raton","hueco"])}
    filler=["el","la","de","y","en","con","por","un","una","que"]
    seq=[]; tags=[]; kind=[]
    for w,(sa,sb) in poly.items():
        for _ in range(n):
            c=[rng.choice(filler) for _ in range(2)]+list(sa[:2])+[w]+list(sa[1:2]); seq+=c; tags+=[0]+[0]*(len(c)-1); kind+=['A']+['A']*(len(c)-1)
        for _ in range(n):
            c=[rng.choice(filler) for _ in range(2)]+list(sb[:2])+[w]+list(sb[1:2]); seq+=c; tags+=[1]+[0]*(len(c)-1); kind+=['B']+['B']*(len(c)-1)
        for _ in range(n):
            c=[rng.choice(filler) for _ in range(1)]+[w]+list(sa[:1])+list(sb[:1])+[rng.choice(filler)]; seq+=c; tags+=[0]+[0]*(len(c)-1); kind+=['MIX']+['MIX']*(len(c)-1)
    pr=list(zip(seq,tags,kind)); rng.shuffle(pr); seq=[p[0] for p in pr]; tags=[p[1] for p in pr]; kind=[p[2] for p in pr]
    return seq,tags,kind,list(dict.fromkeys(seq)),list(poly.keys()),poly
def train_fractal(seq,vocab,epochs):
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}; rng=random.Random(SEED)
    frac=[[[rng.gauss(0,1) for _ in range(D)] for _ in range(K)] for _ in range(Vn)]
    o0=[[list(o) for o in frac[wi]] for wi in range(Vn)]
    for ep in range(epochs):
        for i in range(1,len(seq)):
            a=idx[seq[i-1]]; b=idx[seq[i]]; tb=frac[b][0]
            for k in range(K):
                new=[(1-BETA)*frac[a][k][d]+BETA*tb[d] for d in range(D)]
                new=[ALPHA*o0[a][k][d]+(1-ALPHA)*new[d] for d in range(D)]
                j=1-k; nj=norm(frac[a][j])
                if nj>1e-9: new=[new[d]-BETA_REP*(frac[a][j][d]/nj) for d in range(D)]
                frac[a][k]=new
    return frac,idx
def ctx_vec(frac,idx,seq,i):
    cw=list(range(max(0,i-WIN),i))
    if not cw: return None
    c=[0.0]*D; n=0
    for x in cw:
        for k in range(K):
            for d in range(D): c[d]+=frac[idx[seq[x]]][k][d]
        n+=K
    return [v/n for v in c] if n else None
def train_W(frac,idx,seq,pw,use_tags=False,tags=None):
    rng=random.Random(SEED)
    W=[[1.0 if i==j else 0.01*rng.gauss(0,1) for j in range(D)] for i in range(D)]
    for ep in range(EPOCHS_W):
        for i,w in enumerate(seq):
            if w not in idx: continue
            c=ctx_vec(frac,idx,seq,i)
            if c is None: continue
            if use_tags and w in set(pw): sense=tags[i]
            else:
                cw=list(range(max(0,i-WIN),i))
                if not cw: continue
                lc=[0.0]*D; n=0
                for x in cw:
                    for k in range(K):
                        for d in range(D): lc[d]+=frac[idx[seq[x]]][k][d]
                    n+=K
                if not n: continue
                lc=[v/n for v in lc]
                sc=[cos(frac[idx[w]][k],lc) for k in range(K)]; sense=max(range(K),key=lambda k:sc[k])
            pc=[v/(norm(c) or 1) for v in c]; ps=frac[idx[w]][sense]; np_=norm(ps)
            if np_>1e-9: ps=[v/np_ for v in ps]
            pc=mat_vec(W,pc); ps=mat_vec(W,ps)
            W=[[W[i][j]+LR_W*pc[i]*ps[j] for j in range(D)] for i in range(D)]
    return W
def route(frac,idx,W,w,c,margin):
    pc=mat_vec(W,c); sc=[cos(mat_vec(W,frac[idx[w]][k]),pc) for k in range(K)]
    o=sorted(range(K),key=lambda k:-sc[k]); return o[0],(sc[o[0]]-sc[o[1]])<margin,round(sc[o[0]],3),round(sc[o[1]],3)
def pct(vals,p):
    s=sorted(v for v in vals if v==v and v is not None)
    if not s: return 0.0
    k=(len(s)-1)*p/100; f=int(k)
def main():
    print("=== v0.22 v5 ROOT DIRECTOR (contextos MIXTOS + proy SUAVE + MARGIN adapt) ===")
    t0=time.time()
    seq,tags,kind,vocab,pw,poly=build_corpus()
    frac,idx=train_fractal(seq,vocab,15)
    W=train_W(frac,idx,seq,pw,use_tags=True,tags=tags)
    # diffs para margin adaptativo (percentil PCT sobre todo el corpus)
    diffs=[]
    for i,w in enumerate(seq):
        if w not in set(pw): continue
        c=ctx_vec(frac,idx,seq,i)
        if c is None: continue
        _,_,s1,s2=route(frac,idx,W,w,c,1.0); diffs.append(s1-s2)
    m=pct(diffs,PCT)
    print(f"MARGIN adaptativo (p{PCT}) = {round(m,3)}")
    # duda por tipo de contexto
    counts={"A":[0,0],"B":[0,0],"MIX":[0,0]}
    for i,w in enumerate(seq):
        if w not in set(pw): continue
        c=ctx_vec(frac,idx,seq,i)
        if c is None: continue
        _,d,_,_=route(frac,idx,W,w,c,m)
        counts[kind[i]][1]+=1
        if d: counts[kind[i]][0]+=1
    res={k:dict(duda=round(v[0]/v[1],3) if v[1] else 0, n=v[1]) for k,v in counts.items()}
    print(f"train+eval {time.time()-t0:.0f}s")
    print("DUDA por tipo de contexto (A=univocoA, B=univocoB, MIX=ambos):", res)
    hip = res["MIX"]["duda"] > res["A"]["duda"] and res["MIX"]["duda"] > res["B"]["duda"]
    out=dict(experiment="v0.22_v5_contextos_mixtos",
             hypothesis="Con proyeccion suave + contextos mixtos, la duda DEBE emerger en MIX y no en A/B univocos.",
             params=dict(d=D,window=WIN,k=K,beta=BETA,beta_rep=BETA_REP,alpha=ALPHA,lr_w=LR_W,epochs_w=EPOCHS_W,pct=PCT),
             margin_adaptativo=round(m,3), duda_por_tipo=res,
             duda_emerge_en_mixto=hip)
    json.dump(out,open("results_v22_v5.json","w"),indent=2)
    print("\n-> results_v22_v5.json")
if __name__=="__main__": main()

