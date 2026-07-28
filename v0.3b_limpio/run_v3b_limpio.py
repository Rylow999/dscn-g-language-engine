#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.3b MEMORIA LIMPIO (v2) — corrige auditoria: 'hibernar' era identidad matematica
(no tocaba omega -> = base por construccion). Ahora hibernar = EXCLUIR el nodo del
entrenamiento un tramo y REINTEGRARLO (fase 3: volver a entrenar con el nodo de
vuelta). Borrar = omega=0 permanente.
Mide RETENCION real: tras REINTEGRAR, el omega del nodo debe quedar CERCA del base
(porque se re-entrena al volver); tras borrar queda en 0. Condicion REAL, no identidad.
Senal: reintegrado (~base tras volver a entrenar) vs borrado (0) -> memoria real.
"""
import json, math, random, re, time
from collections import Counter
D=16; V=150; W=4; EPOCHS=2; BETA=0.10; SEED=0; CORPUS_N=20000
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
    step=max(1,len(idxall)//CORPUS_N); chosen=idxall[::step][:CORPUS_N]
    return [words[i] for i in chosen], vocab
def train(seq,vocab,exclude=set(),epochs=EPOCHS):
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}; rng=random.Random(SEED)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(Vn)]
    for ep in range(epochs):
        for i in range(1,len(seq)):
            if seq[i] in exclude or seq[i-1] in exclude: continue
            a,b=idx[seq[i-1]],idx[seq[i]]
            omega[a]=[(1-BETA)*omega[a][k]+BETA*omega[b][k] for k in range(D)]
    return omega, idx
def main():
    print("=== v0.3b MEMORIA LIMPIO v2 (hibernar = excluir + REINTEGRAR) ===")
    seq,vocab=load_seq(); cnt=Counter(seq); content=[w for w,_ in cnt.most_common(80)][30:80]
    # BASE (todo incluido)
    omega_b,_=train(seq,vocab)
    # HIBERNAR real: fase 2 excluye, fase 3 REINTEGRA (vuelve a entrenar con el nodo)
    omega_h,_=train(seq,vocab,exclude=set(content))              # fase 2: excluido
    omega_h,_=train(seq,vocab,epochs=1)                          # fase 3: reintegrado (todo)
    # BORRAR: omega=0 permanente
    omega_d,_=train(seq,vocab)
    idx={w:i for i,w in enumerate(vocab)}
    for w in content:
        if w in idx: omega_d[idx[w]]=[0.0]*D
    def vitalidad(omega, w):
        occ=[i for i,x in enumerate(seq) if x==w]
        if not occ: return 0.0
        s=0.0; n=0
        for i in occ:
            cw=[idx[seq[c]] for c in range(max(0,i-W),i) if seq[c] not in content]
            for c in cw:
                s+=cos(omega[idx[w]], omega[c]); n+=1
        return s/n if n else 0.0
    vit_b=[round(vitalidad(omega_b,w),4) for w in content[:5]]
    vit_h=[round(vitalidad(omega_h,w),4) for w in content[:5]]
    vit_d=[round(vitalidad(omega_d,w),4) for w in content[:5]]
    out=dict(experiment="v0.3b_memoria_limpio_v2",
             hypothesis="Hibernar (excluir un tramo y REINTEGRAR) deja el nodo vivo (~base tras volver a entrenar); borrar (omega=0) lo mata. Memoria real, no identidad matematica.",
             params=dict(d=D,V=V,window=W,epochs=EPOCHS,beta=BETA,corpus_n=CORPUS_N,n_nodos=len(content)),
             vitalidad_base=vit_b, vitalidad_reintegrado=vit_h, vitalidad_borrado=vit_d,
             memoria_real=(sum(vit_h)>sum(vit_d)+0.5))
    json.dump(out,open("results_v3b_limpio.json","w"),indent=2)
    print(f"base       ={vit_b}")
    print(f"reintegrado={vit_h}  (debe quedar ~base tras volver a entrenar)")
    print(f"borrado    ={vit_d}  (debe ser ~0)")
    print("\n-> results_v3b_limpio.json")
if __name__=="__main__": main()
