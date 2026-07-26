#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.9a-bis v2 — DOLOR acoplado a GENERACION que SI repite (generador v0.5b).
El generador v0.5b (afinidad pura, sin top-k) REPITE (loop "el casa el casa").
Ahi el dolor TIENE donde actuar: si genera repeticion adyacente, aplica evasion
(omega del nodo generador se aleja del repetido) y reelije. Mide repeticion
con y sin dolor en GENERACION real.
"""
import json, math, random, re, sys, time
D=8; ALPHA=5.0; BETA=0.20; V=150; STEPS=4000; SEED=0

def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words, V):
    from collections import Counter
    return [w for w,_ in Counter(words).most_common(V)]
def omega_of(w):
    r=random.Random(hash(w)%100000); return [r.gauss(0,1) for _ in range(D)]
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    if na<1e-9 or nb<1e-9: return 0.0
    return sum(x*y for x,y in zip(a,b))/(na*nb)

def generate(omega, idx, vocab, start, steps_gen=60, apply_pain=False):
    out=[start]
    for _ in range(steps_gen):
        cur=idx[out[-1]]
        cands=[(cos(omega[j],omega[cur]), j) for w,j in idx.items() if w!=out[-1]]
        cands.sort(reverse=True)
        best_j=cands[0][1]
        best_w=vocab[best_j]
        if best_w==out[-1] or (len(out)>1 and best_w==out[-2]):
            # dolor: repeticion detectada
            if apply_pain:
                # evasion: omega[cur] se aleja de omega[best_j]
                cb=cos(omega[cur],omega[best_j])
                omega[cur]=[omega[cur][k]-0.2*omega[best_j][k] for k in range(D)]
                # reelije el 2do mejor
                best_j=cands[1][1]; best_w=vocab[best_j]
        out.append(best_w)
    return out

def main():
    print("=== v0.9a-bis v2 DOLOR en generacion que repite (v0.5b) ===")
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text); vocab=build_vocab(words,V)
    rng=random.Random(SEED)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    seq=[w for w in words if w in set(vocab)]
    idx={w:i for i,w in enumerate(vocab)}
    t0=time.time()
    for i in range(1,len(seq)):
        a,b=seq[i-1],seq[i]; ia,ib=idx[a],idx[b]
        omega[ia]=[(1-BETA)*omega[ia][k]+BETA*omega[ib][k] for k in range(D)]
    print(f"entrenado {time.time()-t0:.0f}s")
    # SIN dolor
    g_nopain=generate(omega, idx, vocab, "el", apply_pain=False)
    rep_nopain=sum(1 for i in range(1,len(g_nopain)) if g_nopain[i]==g_nopain[i-1])/(len(g_nopain)-1)
    # CON dolor (usa la MISMA omega, aplica evasion online)
    g_pain=generate([row[:] for row in omega], idx, vocab, "el", apply_pain=True)
    rep_pain=sum(1 for i in range(1,len(g_pain)) if g_pain[i]==g_pain[i-1])/(len(g_pain)-1)
    out=dict(experiment="v0.9a_bis_v2_dolor_generacion",
             definicion_dolor="senal que obliga al sistema a cambiar para evitar lo que lo produce (Luciano)",
             hypothesis="Dolor acoplado a generacion que repite (evasion omega online) reduce repeticion en lo que el grafo HABLA.",
             params=dict(d=D,alpha=ALPHA,beta=BETA,V=V,steps_gen=60),
             repeticion_sin_dolor=round(rep_nopain,4),
             repeticion_con_dolor=round(rep_pain,4),
             mejora=round(rep_nopain-rep_pain,4),
             nota="Generador v0.5b (afinidad pura, SI repite). Dolor online reelije 2do mejor + evasion omega.")
    with open("results_v09abis.json","w") as f: json.dump(out,f,indent=2)
    print(f"rep SIN dolor: {rep_nopain:.4f}  CON dolor: {rep_pain:.4f}  mejora: {rep_nopain-rep_pain:.4f}")
    print("\n-> results_v09abis.json")

if __name__=="__main__": main()
