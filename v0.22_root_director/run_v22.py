#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.22 v3 — ROOT DIRECTOR con PROYECCION Hebb (sin backprop).
v0.22 v1/v2: routing_acc ~0.5 (azar) con contexto promedio O ganadores. El problema
no es el agregado: es que en D=16 el coseno plano no separa los sentidos por contexto.
FIX (raiz): PROYECCION W aprendida por Hebb (SIN backprop, perfil DSCN-G) para que
el contexto relevante caiga cerca del subnodo correcto. Igual que Wq/Wk/Wv del
transformer pero por asociacion local, no gradiente.
  - W (DxD) entrenada: para cada (contexto, sentido_correcto) refuerza W tal que
    cos(W*ctx, W*subnodo_correcto) suba. Hebb: W += lr * (W*ctx) (W*subnodo)^T
  - Ruteo: k* = argmax_k cos(W*frac[w][k], W*ctx)
Hipotesis: con proyeccion, routing_acc sube > 0.5 (real progreso, ataca la raiz).
"""
import json, math, random, re, time
from collections import Counter
D=16; WIN=4; K=2; BETA=0.10; BETA_REP=0.20; ALPHA=0.10; LR_W=0.01; EPOCHS_W=3
MARGINS=[0.05,0.10,0.15,0.20]; SEED=0
def norm(v): return math.sqrt(sum(x*x for x in v))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def mat_vec(M,v): return [sum(M[i][j]*v[j] for j in range(D)) for i in range(D)]
def build_contrast_corpus(seed=SEED, n_per_sense=50):
    rng=random.Random(seed)
    poly={
      "banco": (["dinero","pagar","cuenta","oro","plata"], ["rio","agua","pez","orilla","puente"]),
      "llave": (["puerta","cerradura","abrir","candado"], ["musica","nota","tono","cancion"]),
      "mouse": (["computadora","click","pantalla","cable"], ["animal","cola","raton","hueco"]),
    }
    filler=["el","la","de","y","en","con","por","un","una","que","los","las"]
    seq=[]; tags=[]
    for w,(sa,sb) in poly.items():
        for _ in range(n_per_sense):
            ctx=[rng.choice(filler) for _ in range(3)] + list(sa[:3]) + [w] + list(sa[1:3])
            seq+=ctx; tags+=[0]+[0]*(len(ctx)-1)
        for _ in range(n_per_sense):
            ctx=[rng.choice(filler) for _ in range(3)] + list(sb[:3]) + [w] + list(sb[1:3])
            seq+=ctx; tags+=[1]+[0]*(len(ctx)-1)
    paired=list(zip(seq,tags)); rng.shuffle(paired); seq=[p[0] for p in paired]; tags=[p[1] for p in paired]
    vocab=list(dict.fromkeys(seq))
    return seq, tags, vocab, list(poly.keys()), poly
def train_fractal(seq,vocab,epochs):
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}; rng=random.Random(SEED)
    frac=[[[rng.gauss(0,1) for _ in range(D)] for _ in range(K)] for _ in range(Vn)]
    omega0=[[list(o) for o in frac[wi]] for wi in range(Vn)]
    for ep in range(epochs):
        for i in range(1,len(seq)):
            a=idx[seq[i-1]]; b=idx[seq[i]]; tb=frac[b][0]
            for k in range(K):
                new=[(1-BETA)*frac[a][k][d]+BETA*tb[d] for d in range(D)]
                new=[ALPHA*omega0[a][k][d]+(1-ALPHA)*new[d] for d in range(D)]
                j=1-k; nj=norm(frac[a][j])
                if nj>1e-9: new=[new[d]-BETA_REP*(frac[a][j][d]/nj) for d in range(D)]
                frac[a][k]=new
    return frac, idx
def context_vec(frac,idx,seq,i):
    cw=list(range(max(0,i-WIN),i))
    if not cw: return None
    ctx=[0.0]*D; n=0
    for c in cw:
        for k in range(K):
            for d in range(D): ctx[d]+=frac[idx[seq[c]]][k][d]
        n+=K
    return [x/n for x in ctx] if n else None
def train_proj(frac,idx,seq,tags,poly_words):
    rng=random.Random(SEED)
    W=[[1.0 if i==j else 0.01*rng.gauss(0,1) for j in range(D)] for i in range(D)]
    for ep in range(EPOCHS_W):
        for i,w in enumerate(seq):
            if w not in poly_words: continue
            ctx=context_vec(frac,idx,seq,i)
            if ctx is None: continue
            sense=tags[i]
            # Hebb: refuerza W tal que W*ctx sea afín a W*subnodo[sense]
            pctx=[x/ (norm(ctx) or 1) for x in ctx]
            psub=frac[idx[w]][sense]; np_=norm(psub)
            if np_>1e-9: psub=[x/np_ for x in psub]
            pc=mat_vec(W,pctx); ps=mat_vec(W,psub)
            # W += lr * outer(pc, ps)  (Hebb asociativo)
            W=[[W[i][j]+LR_W*pc[i]*ps[j] for j in range(D)] for i in range(D)]
    return W
def root_route(frac,idx,W,w,ctx,margin):
    pc=mat_vec(W,ctx)
    sc=[cos(mat_vec(W,frac[idx[w]][k]),pc) for k in range(K)]
    order=sorted(range(K), key=lambda k:-sc[k])
    s1,s2=sc[order[0]],sc[order[1]]
    doubt=(s1-s2)<margin
    return order[0], doubt, round(s1,3), round(s2,3)
def phase_A(frac,idx,W,seq,tags,poly_words,poly):
    res={}
    for margin in MARGINS:
        ok=tot=0
        for i,w in enumerate(seq):
            if w not in poly_words: continue
            ctx=context_vec(frac,idx,seq,i)
            if ctx is None: continue
            routed,doubt,_,_=root_route(frac,idx,W,w,ctx,margin)
            tot+=1
            if not doubt and routed==tags[i]: ok+=1
        res[str(margin)]=dict(routing_acc=round(ok/tot,3) if tot else 0, total=tot)
    probe={}
    for margin in MARGINS:
        dudas=0; n=0
        for w,(sa,sb) in poly.items():
            ca=context_words(frac,idx,sa); cb=context_words(frac,idx,sb)
            if ca is None or cb is None: continue
            ambig=[(ca[d]+cb[d])/2 for d in range(D)]
            _,doubt,_,_=root_route(frac,idx,W,w,ambig,margin)
            dudas+=1 if doubt else 0; n+=1
        probe[str(margin)]=dict(duda_ambigua=dudas, n=n)
    return res, probe
def context_words(frac,idx,words):
    vecs=[]
    for w in words:
        if w not in idx: continue
        for k in range(K):
            v=[x/ (norm(frac[idx[w]][k]) or 1) for x in frac[idx[w]][k]]; vecs.append(v)
    if not vecs: return None
    out=[0.0]*D
    for v in vecs:
        for d in range(D): out[d]+=v[d]
    return [x/len(vecs) for x in out]
def phase_B(frac,idx,W):
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=re.findall(r"[a-záéíóúñü]+", text.lower())
    vocab=[w for w,_ in Counter(words).most_common(150)]
    rng=random.Random(SEED)
    idxall=[i for i,w in enumerate(words) if w in set(vocab)]
    step=max(1,len(idxall)//20000); chosen=idxall[::step][:20000]
    seq=[words[i] for i in chosen]
    # grafo + proyeccion W propios de Don Quijote (idx distinto al contrastivo)
    fracB,idxB=train_fractal(seq,vocab,epochs=4)
    # W_B entrenada sobre Don Quijote: pseudo-ground-truth = subnodo ganador local
    # (ruteo por contexto del grafo fractal ya separado). Hebb asociativo.
    WB=train_proj_dq(fracB,idxB,seq)
    cnt=Counter(seq); cand=[w for w in vocab if cnt[w]>=20][:40]
    res={}
    for margin in MARGINS:
        doubt=0; tot=0
        for i,w in enumerate(seq):
            if w not in cand: continue
            ctx=context_vec(fracB,idxB,seq,i)
            if ctx is None: continue
            _,d,_,_=root_route(fracB,idxB,WB,w,ctx,margin)
            if d: doubt+=1
            tot+=1
        res[str(margin)]=dict(tasa_duda=round(doubt/tot,3) if tot else 0, total=tot)
    return res
def train_proj_dq(frac,idx,seq):
    rng=random.Random(SEED)
    W=[[1.0 if i==j else 0.01*rng.gauss(0,1) for j in range(D)] for i in range(D)]
    for ep in range(EPOCHS_W):
        for i,w in enumerate(seq):
            if w not in idx: continue
            ctx=context_vec(frac,idx,seq,i)
            if ctx is None: continue
            # pseudo-ground-truth: subnodo ganador local del grafo fractal
            cw=list(range(max(0,i-WIN),i))
            if not cw: continue
            lctx=[0.0]*D; n=0
            for c in cw:
                for k in range(K):
                    for d in range(D): lctx[d]+=frac[idx[seq[c]]][k][d]
                n+=K
            if not n: continue
            lctx=[x/n for x in lctx]
            sc=[cos(frac[idx[w]][k],lctx) for k in range(K)]
            sense=max(range(K), key=lambda k:sc[k])
            pctx=[x/(norm(ctx) or 1) for x in ctx]
            psub=frac[idx[w]][sense]; nps=norm(psub)
            if nps>1e-9: psub=[x/nps for x in psub]
            pc=mat_vec(W,pctx); ps=mat_vec(W,psub)
            W=[[W[i][j]+LR_W*pc[i]*ps[j] for j in range(D)] for i in range(D)]
    return W
def main():
    print("=== v0.22 v3 ROOT DIRECTOR (PROYECCION Hebb, sin backprop) ===")
    print(">> ataca la raiz: contexto plano no separa sentidos en D=16; W Hebb si")
    t0=time.time()
    seq,tags,vocab,poly_words,poly=build_contrast_corpus()
    frac,idx=train_fractal(seq,vocab,epochs=15)
    W=train_proj(frac,idx,seq,tags,poly_words)
    resA, probeA = phase_A(frac,idx,W,seq,tags,poly_words,poly)
    resB = phase_B(frac,idx,W)
    print(f"train+eval {time.time()-t0:.0f}s")
    print("FASE A routing_acc por margin:", resA)
    print("FASE A probe ambiguo duda:", probeA)
    print("FASE B tasa_duda por margin:", resB)
    out=dict(experiment="v0.22_v3_root_director_proyeccion_hebb",
             hypothesis="Proyeccion W Hebb (sin backprop) hace que el contexto relevante caiga cerca del subnodo correcto -> routing_acc sube > 0.5 (raiz, no agregado).",
             params=dict(d=D,window=W,k=K,beta=BETA,beta_rep=BETA_REP,alpha=ALPHA,lr_w=LR_W,epochs_w=EPOCHS_W,margins=MARGINS),
             fase_A_routing=resA, fase_A_probe_ambiguo=probeA, fase_B_donquijote=resB,
             mejor_acc=max(resA[m]["routing_acc"] for m in resA),
             mejora_sobre_azar=any(resA[m]["routing_acc"]>0.6 for m in resA))
    json.dump(out,open("results_v22.json","w"),indent=2)
    print("\n-> results_v22.json")
if __name__=="__main__": main()
