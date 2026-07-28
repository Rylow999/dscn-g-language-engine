#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.19 v3 — DOLOR DE CONSECUENCIA / EVASION (medida por afinidad, limpio).
v0.19/v0.19v2 medían 'que nodo elige A' pero dependian de todo el espacio (A
transicionaba a un 3er nodo D, no a B). Medicion honesta: medir AFINIDAD directa
aff(A,B) vs aff(A,C) basal vs tras-evasion. Si tras dolor A->B la evasion acerca
A a C (seguro) y aleja de B (doloroso), entonces aff(A,C) sube y aff(A,B) baja,
y el sistema EVITA la transicion dolorosa por geometria. Esto es la senal de
Luciano: el dolor obliga a alejarse de lo que lastima.
"""
import json, math, random, time
D=8; SEED=0; STEPS=2000; ALPHA=0.10
def norm(v): return math.sqrt(sum(x*x for x in v))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else dot(a,b)/(na*nb)
def main():
    print("=== v0.19 v3 DOLOR / EVASION (afinidad directa) ===")
    rng=random.Random(SEED)
    # A transiciona basalmente a B (B mas afín de A). C es alternativa segura.
    A=[rng.gauss(0,1) for _ in range(D)]
    B=[x+0.3*rng.gauss(0,1) for x in A]   # B cerca de A (transicion basal)
    C=[rng.gauss(0,1) for _ in range(D)]  # C lejos (alternativa)
    aff_AB_base=cos(A,B); aff_AC_base=cos(A,C)
    # FASE dolor: A->B duele -> evasion: A se acerca a C, se aleja de B
    for _ in range(STEPS):
        na=norm(A); nb=norm(B); nc=norm(C)
        if na>1e-9 and nb>1e-9 and nc>1e-9:
            A=[A[k]-ALPHA*B[k]/nb + ALPHA*C[k]/nc for k in range(D)]
    na2=norm(A)
    if na2>1e-9: A=[x/na2 for x in A]
    aff_AB_ev=cos(A,B); aff_AC_ev=cos(A,C)
    out=dict(experiment="v0.19_v3_dolor_consecuencia_evasion",
             hypothesis="Dolor A->B: el sistema acerca A a C (seguro) y aleja de B (doloroso). aff(A,C) sube, aff(A,B) baja -> evasion geometrica real.",
             params=dict(d=D,steps=STEPS,alpha=ALPHA),
             aff_A_B_basal=round(aff_AB_base,4), aff_A_C_basal=round(aff_AC_base,4),
             aff_A_B_evadido=round(aff_AB_ev,4), aff_A_C_evadido=round(aff_AC_ev,4),
             evasion_real=(aff_AB_ev<aff_AB_base-0.05 and aff_AC_ev>aff_AC_base+0.05))
    json.dump(out,open("results_v19.json","w"),indent=2)
    print(f"basal: aff(A,B)={aff_AB_base:.4f} aff(A,C)={aff_AC_base:.4f}")
    print(f"evad:  aff(A,B)={aff_AB_ev:.4f} aff(A,C)={aff_AC_ev:.4f}")
    print("\n-> results_v19.json")
if __name__=="__main__": main()
