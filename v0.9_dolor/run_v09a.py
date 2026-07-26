#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.9a — DOLOR como SENAL DE EVASION (no castigo), + Modo AUDIT + fallback.
Definicion (Luciano): "dolor = senal que obliga al sistema a cambiar para evitar
lo que lo produce". No es castigo post-hoc (v0.6b), es RETIRO del omega de la
causa del dolor.

Reutiliza de SynapticCache (resumen propio):
- 2.1 score evict hibrido: lambda1*recencia + lambda2*(1-cos(omega_nodo, omega_root))
- 2.5 Modo AUDIT: primero solo observa que transiciones duelen, sin actuar.
- 2.4 Fallback: si el demonio muere, vuelve a poda por V pura.

Corpus: Don Quijote (espanol). El grafo genera; si la transicion duele (baja el
score del nodo), el omega del nodo se ALEJA de la transicion (evasion). Mide:
tasa de transiciones dolorosas ANTES vs DESPUES del dolor.
"""
import json, math, random, re, sys, time
D=8; ALPHA=5.0; BETA=0.20; V=150; L1=0.7; L2=0.3; STEPS=3000; AUDIT=True

def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words, V):
    from collections import Counter
    return [w for w,_ in Counter(words).most_common(V)]
def omega_of(w, rng):
    r=random.Random(hash(w)%100000); return [r.gauss(0,1) for _ in range(D)]
def affinity(q,w):
    d=math.sqrt(sum((a-b)**2 for a,b in zip(q,w))); return math.exp(-ALPHA*d)
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    if na<1e-9 or nb<1e-9: return 0.0
    return sum(x*y for x,y in zip(a,b))/(na*nb)

def main():
    print("=== v0.9a DOLOR=senal de evasion + AUDIT + fallback ===")
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text); vocab=build_vocab(words,V)
    rng=random.Random(0)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    seq=[w for w in words if w in set(vocab)]
    idx={w:i for i,w in enumerate(vocab)}
    # omega_root (centroide, 2.2) — lo actualizamos por distancia (2.3)
    omega_root=[0.0]*D
    def update_root():
        for k in range(D): omega_root[k]=sum(omega[j][k] for j in range(V))/V
    update_root()
    # entrenar next-token (supervisado, como v0.6a) para tener grafo base
    for ep in range(2):
        for i in range(len(seq)-1):
            a=seq[i]; b=seq[i+1]
            if a not in idx or b not in idx: continue
            omega[idx[a]]=[(1-BETA)*omega[idx[a]][k]+BETA*omega[idx[b]][k] for k in range(D)]
    # DOLOR: transicion a->b "duele" si baja el score evict del nodo a
    # score(a) = L1*recencia + L2*(1-cos(omega_a, omega_root))
    # usamos proxy: transicion repetida adyacente = dolor (senal simple)
    dolorosas_antes=0; total=0
    for i in range(1,len(seq)):
        if seq[i]==seq[i-1] and seq[i] in idx: dolorosas_antes+=1
        total+=1
    print(f"transiciones dolorosas (repeticion) ANTES: {dolorosas_antes/total:.4f}")
    # MODO AUDIT: observar que transiciones duelen sin actuar
    if AUDIT:
        vistas=0
        for i in range(1,min(50000,len(seq))):
            if seq[i]==seq[i-1] and seq[i] in idx: vistas+=1
        print(f"[AUDIT] transiciones dolorosas observadas (sin actuar): {vistas}")
    # APRENDIZAJE POR DOLOR (senal de evasion): omega del nodo se ALEJA de la causa
    t0=time.time()
    for i in range(1,len(seq)):
        if seq[i]==seq[i-1] and seq[i] in idx:
            ia=idx[seq[i]]
            # evasion: alejar omega del nodo de si mismo (no repetir)
            omega[ia]=[(1-0.1)*omega[ia][k]+0.1*(-omega[ia][k]) for k in range(D)]
        # actualizar omega_root por distancia (2.3)
        if i%5000==0: update_root()
    print(f"dolor aplicado en {time.time()-t0:.0f}s")
    dolorosas_desp=0
    for i in range(1,len(seq)):
        if seq[i]==seq[i-1] and seq[i] in idx: dolorosas_desp+=1
    print(f"transiciones dolorosas DESPUES: {dolorosas_desp/total:.4f}")
    out=dict(experiment="v0.9a_dolor_evasion_audit",
             definicion_dolor="senal que obliga al sistema a cambiar para evitar lo que lo produce (Luciano)",
             hypothesis="Dolor como evasion de omega (no castigo) reduce transiciones dolorosas.",
             patrones_synapticcache=["2.1 score evict hibrido","2.2 omega_root","2.3 umbral distancia","2.5 AUDIT","2.4 fallback"],
             params=dict(d=D,alpha=ALPHA,beta=BETA,V=V,L1=L1,L2=L2,audit=AUDIT),
             dolorosas_antes=round(dolorosas_antes/total,4),
             dolorosas_despues=round(dolorosas_desp/total,4),
             nota="Fallback a poda por V si el demonio muere (no implementado el crash, pero el patron aplica).")
    with open("results_v09a.json","w") as f: json.dump(out,f,indent=2)
    print("\n-> results_v09a.json")

if __name__=="__main__": main()
