#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.21 v8 REAL — ANCHOR+REPULSION sobre Don Quijote (validar fix oversmoothing
en corpus REAL, no sintetico). Misma regla que v0.21 v8 pero sobre Don Quijote
(top-150 vocab, 20k tok). Test desambig: palabras con 2 sentidos separados y
ESTABLES a lo largo de epocas (no transitorio). Si >0 estable -> el fix es
general, no solo en corpus controlado.
"""
import json, math, random, re, time
from collections import Counter
D=16; V=150; W=4; EPOCHS=8; K=2; BETA=0.10; BETA_REP=0.20; ALPHA=0.10
SEED=0
def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def norm(v): return math.sqrt(sum(x*x for x in v))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def load_seq():
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text); vocab=[w for w,_ in Counter(words).most_common(V)]
    rng=random.Random(SEED)
    idxall=[i for i,w in enumerate(words) if w in set(vocab)]
    step=max(1,len(idxall)//20000); chosen=idxall[::step][:20000]
    return [words[i] for i in chosen], vocab
def main():
    print("=== v0.21 v8 REAL (Don Quijote, anchor+repulsion) ===")
    seq,vocab=load_seq(); Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    frac=[[[rng.gauss(0,1) for _ in range(D)] for _ in range(K)] for _ in range(Vn)]
    omega0=[[list(o) for o in frac[wi]] for wi in range(Vn)]
    t0=time.time()
    estables_por_ep=[]
    for ep in range(EPOCHS):
        for i in range(1,len(seq)):
            a=idx[seq[i-1]]; b=idx[seq[i]]; tb=frac[b][0]
            for k in range(K):
                new=[(1-BETA)*frac[a][k][d]+BETA*tb[d] for d in range(D)]
                new=[ALPHA*omega0[a][k][d]+(1-ALPHA)*new[d] for d in range(D)]
                j=1-k; nj=norm(frac[a][j])
                if nj>1e-9: new=[new[d]-BETA_REP*(frac[a][j][d]/nj) for d in range(D)]
                frac[a][k]=new
        # medir desambig (mismo test que v0.21 v3/v4)
        cnt=Counter(seq); cand=[w for w in vocab if cnt[w]>=20]
        ok=0
        for w in cand[:40]:
            occ=[i for i,x in enumerate(seq) if x==w]; grupos={}
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
            if len(grupos)>=2 and max(grupos.values())<len(occ)*0.85: ok+=1
        estables_por_ep.append((ep+1,ok))
    print(f"train {time.time()-t0:.0f}s")
    for e,ok in estables_por_ep: print(f"  ep{e}: {ok}/40")
    out=dict(experiment="v0.21_v8_real_donquijote",
             hypothesis="Anchor+repulsion sostiene separacion de sentidos en Don Quijote real (no solo sintetico). Fix oversmoothing es general.",
             params=dict(d=D,V=V,window=W,epochs=EPOCHS,k=K,beta=BETA,beta_rep=BETA_REP,alpha=ALPHA),
             curva=[{"epoca":e,"separadas":ok} for e,ok in estables_por_ep],
             ultima=estables_por_ep[-1][1],
             veredicto=("SEPARACION ESTABLE EN CORPUS REAL" if estables_por_ep[-1][1]>0 else "aun no en real"))
    json.dump(out,open("results_v21_v8_real.json","w"),indent=2)
    print("\n-> results_v21_v8_real.json")
if __name__=="__main__": main()
