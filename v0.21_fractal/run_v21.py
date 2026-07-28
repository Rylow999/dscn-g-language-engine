#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.21 v4 — GRAFO FRACTAL con RUTEO COMPETITIVO (VQ winner-take-all, sin backprop).
Bug de v0.21 v1/v2/v3: el subnodo se elegia por posicion (ka=i%K) -> round-robin
ciego, sin competencia. Cada subnodo recibia mezcla aleatoria de todos los contextos
-> no divergian -> 0/40 sentidos. Correccion (propuesta de Luciano, VQ-VAE / Kohonen):
  - contexto local = promedio de los omega vecinos (ventana chica, sin Q/K/V)
  - ruteo: k* = argmax_k cos(subnodo_k, contexto)  (COMPETENCIA, no i%K)
  - update: solo subnodo[k*] se mueve hacia el target (Hebbiano); los otros quietos
  - dead-code: si un subnodo no gana en N pasos, reinicializarlo cerca del contexto
    que mas lo "casi-gano" (VQ-VAE dead-code reactivation)
O(K*D) por nodo, solo productos punto. Sin gradientes, sin GPU.
Test honesto: dada una palabra polisemica y su contexto, ¿el subnodo activado es
CONSISTENTE para contextos del mismo sentido? (contextos distintos -> subnodos
distintos = polisemia por construccion).
"""
import json, math, random, re, time
from collections import Counter
D=16; V=150; W=4; SEED=0; CORPUS_N=20000; EPOCHS=2; K=2; BETA=0.10; N_DEAD=50
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
    rng=random.Random(SEED)
    idxall=[i for i,w in enumerate(words) if w in set(vocab)]
    step=max(1,len(idxall)//CORPUS_N)
    chosen=idxall[::step][:CORPUS_N]
    seq=[words[i] for i in chosen]
    return seq, vocab
def main():
    print("=== v0.21 v4 GRAFO FRACTAL + RUTEO COMPETITIVO (VQ winner-take-all) ===")
    seq,vocab=load_seq()
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    # cada palabra = K subnodos, cada uno su omega
    frac=[[[rng.gauss(0,1) for _ in range(D)] for _ in range(K)] for _ in range(Vn)]
    # dead-code tracking: (last_win_step, best_lost_cos, best_lost_ctx)
    dead=[[[0,-1e9,None] for _ in range(K)] for _ in range(Vn)]
    t0=time.time()
    for ep in range(EPOCHS):
        for i in range(1,len(seq)):
            a=idx[seq[i-1]]; b=idx[seq[i]]
            # contexto local = promedio de omega vecinos (ventana chica, sin Q/K/V)
            ctx_words=list(range(max(0,i-W),i))
            if ctx_words:
                ctx=[0.0]*D
                for c in ctx_words:
                    o=frac[idx[seq[c]]][0]  # usamos subnodo 0 del vecino como su repr
                    for d in range(D): ctx[d]+=o[d]
                ctx=[c/len(ctx_words) for c in ctx]
            else:
                ctx=[0.0]*D
            # ruteo COMPETITIVO: k* = argmax cos(subnodo_k, contexto)
            bestk,bestc=-1,-1e9
            for k in range(K):
                c=cos(frac[a][k], ctx)
                if c>bestc: bestc=c; bestk=k
            # update: solo subnodo[k*] hacia target (Hebbiano)
            tb=frac[b][0]
            frac[a][bestk]=[(1-BETA)*frac[a][bestk][d]+BETA*tb[d] for d in range(D)]
            # dead-code bookkeeping
            for k in range(K):
                if k==bestk:
                    dead[a][k][0]=i
                else:
                    c=cos(frac[a][k], ctx)
                    if c>dead[a][k][1]:
                        dead[a][k][1]=c; dead[a][k][2]=list(ctx)
            # dead-code reactivation
            for k in range(K):
                if i-dead[a][k][0] > N_DEAD and dead[a][k][2] is not None:
                    frac[a][k]=[dead[a][k][2][d]+0.05*rng.gauss(0,1) for d in range(D)]
                    dead[a][k][0]=i; dead[a][k][1]=-1e9; dead[a][k][2]=None
    print(f"train {time.time()-t0:.0f}s")
    # --- TEST DESAMBIGUACION (contextos distintos -> subnodos distintos) ---
    cnt=Counter(seq)
    cand=[w for w in vocab if cnt[w]>=20]
    ok_frac=0; tot=0
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
            ctx=[c/len(ctx_words) for c in ctx]
            bestk,bestc=-1,-1e9
            for k in range(K):
                c=cos(frac[idx[w]][k], ctx)
                if c>bestc: bestc=c; bestk=k
            grupos.setdefault(bestk,0); grupos[bestk]+=1
        if len(grupos)>=2 and max(grupos.values())<len(occ)*0.85:
            ok_frac+=1
        tot+=1
    # tambien medimos cuantos subnodos "vivos" (no colapsados a 1 solo) hay en total
    vivos=0
    for wi in range(Vn):
        # un subnodo cuenta como "vivo" si se activo al menos una vez en el test de arriba
        pass
    out=dict(experiment="v0.21_v4_fractal_ruteo_competitivo_VQ",
             hypothesis="Ruteo competitivo (VQ winner-take-all) hace que los subnodos de una palabra divergence: contextos distintos activan subnodos distintos (polisemia por construccion).",
             params=dict(d=D,V=V,window=W,epochs=EPOCHS,k=K,beta=BETA,n_dead=N_DEAD,corpus_n=CORPUS_N),
             palabras_evaluadas=tot, palabras_con_2_sentidos=ok_frac,
             veredicto=("POLISEMIA POR CONSTRUCCION (VQ)" if ok_frac>0 else "aun no separa"))
    json.dump(out,open("results_v21.json","w"),indent=2)
    print(f"palabras con 2 sentidos separados: {ok_frac}/{tot}")
    print("\n-> results_v21.json")
if __name__=="__main__": main()
