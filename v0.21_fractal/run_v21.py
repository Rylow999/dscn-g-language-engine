#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.21 v5 — GRAFO FRACTAL + COMPETENCIA SUAVE (temperatura, no WTA duro).
Hipotesis de Luciano: v0.21 v4 colapso al ganador por COLD-START WTA (el primer
subnodo que gana se lleva todo por feedback positivo; el otro queda en 0 y el
dead-code lo reinicia en la misma region -> re-pierde). Fix: COMPETENCIA SUAVE con
temperatura (softmax de las similitudes, tipo GMM-EM / annealing VQ). Desde el
paso 0 es suave: AMBOS subnodos se mueven, pesados por afinidad w_k. Eso evita el
lock-in inicial y deja especializar antes de que la temperatura baje.
  - contexto local = promedio de omega vecinos (ventana chica, sin Q/K/V)
  - ruteo SUAVE: w_k = softmax( cos(subnodo_k, contexto) / T )   (T alta=suave)
  - update: todos los subnodos se mueven, pesados por w_k (Hebbiano suave)
  - T ana dentro del entrenamiento (lineal T_init -> T_min): arranca suave, afila
  - dead-code: si un subnodo no gana en N pasos, reinicializarlo cerca del contexto
    que mas lo "casi-gano"
O(K*D) por nodo, solo productos punto. Sin gradientes, sin GPU.
Test honesto (3 semillas, promediado): contextos distintos -> subnodos distintos?
"""
import json, math, random, re, time
from collections import Counter
D=16; V=150; W=4; EPOCHS=2; K=2; BETA=0.10; N_DEAD=50
T_INIT=0.6; T_MIN=0.05
SEEDS=[0,1,2]
def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words,V):
    return [w for w,_ in Counter(words).most_common(V)]
def norm(v): return math.sqrt(sum(x*x for x in v))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def load_seq():
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text)
    vocab=build_vocab(words,V)
    out={}
    for SEED in SEEDS:
        rng=random.Random(SEED)
        idxall=[i for i,w in enumerate(words) if w in set(vocab)]
        step=max(1,len(idxall)//20000)
        chosen=idxall[::step][:20000]
        seq=[words[i] for i in chosen]
        out[SEED]=(seq,vocab)
    return out
def train_one(seq,vocab,seed):
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(seed)
    frac=[[[rng.gauss(0,1) for _ in range(D)] for _ in range(K)] for _ in range(Vn)]
    dead=[[[0,-1e9,None] for _ in range(K)] for _ in range(Vn)]
    total_steps=EPOCHS*(len(seq)-1); si=0
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
            # COMPETENCIA SUAVE: softmax de similitudes / T
            T=T_MIN+(T_INIT-T_MIN)*(1.0 - si/total_steps)
            sims=[cos(frac[a][k],ctx) for k in range(K)]
            mx=max(sims); ex=[math.exp((s-mx)/max(T,1e-6)) for s in sims]
            Z=sum(ex); w=[e/Z for e in ex]
            tb=frac[b][0]
            for k in range(K):
                frac[a][k]=[(1-BETA*w[k])*frac[a][k][d]+BETA*w[k]*tb[d] for d in range(D)]
            # dead-code tracking (ganador = argmax w)
            wk=max(range(K), key=lambda k:w[k])
            for k in range(K):
                if k==wk:
                    dead[a][k][0]=i
                else:
                    c=cos(frac[a][k],ctx)
                    if c>dead[a][k][1]:
                        dead[a][k][1]=c; dead[a][k][2]=list(ctx)
            for k in range(K):
                if i-dead[a][k][0] > N_DEAD and dead[a][k][2] is not None:
                    frac[a][k]=[dead[a][k][2][d]+0.05*rng.gauss(0,1) for d in range(D)]
                    dead[a][k][0]=i; dead[a][k][1]=-1e9; dead[a][k][2]=None
            si+=1
    return frac, idx
def test_desambig(seq,vocab,frac,idx):
    cnt=Counter(seq)
    cand=[w for w in vocab if cnt[w]>=20]
    ok=0; tot=0
    for w in cand[:40]:
        occ=[i for i,x in enumerate(seq) if x==w]
        grupos={}
        for i in occ:
            ctx_words=list(range(max(0,i-W),i))
            if not ctx_words: continue
            ctx=[0.0]*D
            for c in ctx_words:
                o=frac[idx[seq[c]]][0]
                for d in range(D): ctx[d]+=o[d]
            ctx=[x/len(ctx_words) for x in ctx]
            bestk,bestc=-1,-1e9
            for k in range(K):
                c=cos(frac[idx[w]][k],ctx)
                if c>bestc: bestc=c; bestk=k
            grupos.setdefault(bestk,0); grupos[bestk]+=1
        if len(grupos)>=2 and max(grupos.values())<len(occ)*0.85:
            ok+=1
        tot+=1
    return ok, tot
def main():
    print("=== v0.21 v5 GRAFO FRACTAL + COMPETENCIA SUAVE (temperatura) ===")
    data=load_seq()
    results=[]
    for SEED in SEEDS:
        seq,vocab=data[SEED]
        frac,idx=train_one(seq,vocab,SEED)
        ok,tot=test_desambig(seq,vocab,frac,idx)
        results.append((ok,tot))
        print(f"  seed {SEED}: {ok}/{tot} sentidos separados")
    ok_tot=sum(r[0] for r in results); tot_tot=sum(r[1] for r in results)
    mean_ok=ok_tot/len(results)
    out=dict(experiment="v0.21_v5_fractal_competencia_suave_temperatura",
             hypothesis="Competencia suave (softmax/temperatura) evita cold-start WTA: ambos subnodos se mueven pesados por afinidad -> divergen sin colapsar. Polisemia por construccion en grafo rústico.",
             params=dict(d=D,V=V,window=W,epochs=EPOCHS,k=K,beta=BETA,n_dead=N_DEAD,t_init=T_INIT,t_min=T_MIN,seeds=SEEDS),
             por_semilla=[{"ok":r[0],"tot":r[1]} for r in results],
             palabras_con_2_sentidos_promedio=mean_ok,
             veredicto=("POLISEMIA POR CONSTRUCCION (suave)" if mean_ok>0 else "aun no separa"))
    json.dump(out,open("results_v21.json","w"),indent=2)
    print(f"PROMEDIO: {mean_ok:.1f}/{tot_tot} sentidos separados")
    print("\n-> results_v21.json")
if __name__=="__main__": main()
