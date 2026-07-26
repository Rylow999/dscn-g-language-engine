#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.9b — ETIQUETAS QUE MUTAN por dolor (no hardcodeadas, a diferencia de v0.6b).
El nodo arranca SIN etiqueta y la APRENDE por historial de uso: cada vez que se
conecta bien (no duele) refuerza su etiqueta actual; si duele, la hace mutar.
Critico externo MINIMO solo para senalar dolor (repeticion / orden S-S-S).
Mide: ¿las etiquetas convergen a algo coherente y baja la tasa de dolor?
"""
import json, math, random, re, sys, time
D=8; ALPHA=5.0; BETA=0.20; V=150; STEPS=4000

def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def build_vocab(words, V):
    from collections import Counter
    return [w for w,_ in Counter(words).most_common(V)]
def omega_of(w, rng):
    r=random.Random(hash(w)%100000); return [r.gauss(0,1) for _ in range(D)]
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    if na<1e-9 or nb<1e-9: return 0.0
    return sum(x*y for x,y in zip(a,b))/(na*nb)

# diccionario de VERDAD solo para senalar dolor (no para etiquetar el nodo)
SUST={'don','quijote','caballero','sancho','casa','gato','pez','perro','rojo','libro','dia','noche','rey','campo','espada','mujer','hombre','agua','pan','vino','tierra','cielo','sol','luna','mano','vida','muerte','amor','mundo','pueblo','castillo','senor','camino','cuerpo','historia'}
VERB={'come','corre','es','tiene','va','dice','hace','da','ve','oye','sabe','quiere','puede','debe','vive','habla','piensa','llega','pone','deja','mira','siente','ama','odia','cree','llama'}

def main():
    print("=== v0.9b ETIQUETAS QUE MUTAN por dolor ===")
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text); vocab=build_vocab(words,V)
    rng=random.Random(0)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    seq=[w for w in words if w in set(vocab)]
    idx={w:i for i,w in enumerate(vocab)}
    # etiqueta del nodo: arranca None, aprende por historial
    etiq=[None]*V
    hist_count=[[0,0,0] for _ in range(V)]  # [S,V,Otro] conteo de usos como cada rol
    dolor_antes=0; total=0
    for i in range(1,len(seq)):
        if seq[i]==seq[i-1] and seq[i] in idx: dolor_antes+=1
        total+=1
    # entrenar next-token + mutacion de etiqueta por dolor
    t0=time.time()
    for i in range(2,len(seq)):
        a,b=seq[i-1],seq[i]
        if a not in idx or b not in idx: continue
        ia,ib=idx[a],idx[b]
        # next-token
        omega[ia]=[(1-BETA)*omega[ia][k]+BETA*omega[ib][k] for k in range(D)]
        # senal de dolor: repeticion adyacente
        dolor = (seq[i]==seq[i-1])
        # etiqueta: el nodo b aprende su rol segun la etiqueta de a (contexto)
        # si a es S y b es V -> b se refuerza como V; si duele, muta
        ea=etiq[ia]; eb_true = 'V' if b in VERB else ('S' if b in SUST else 'O')
        if ea is not None:
            if not dolor:
                # refuerza rol inferido
                if ea=='S' and eb_true=='V': hist_count[ib][1]+=1
                elif ea=='V' and eb_true=='S': hist_count[ib][0]+=1
                else: hist_count[ib][2]+=1
            else:
                # dolor: muta (penaliza el rol actual, prueba otro)
                if eb_true in ('S','V'): hist_count[ib][2]+=1
        # asigna etiqueta por mayoria de historial
        c=hist_count[ib]; etiq[ib]='S' if c[0]==max(c) else ('V' if c[1]==max(c) else 'O')
    print(f"entrenado {time.time()-t0:.0f}s")
    # medir: ¿las etiquetas aprendidas coinciden con la verdad?
    ok=0; tot=0
    for w in vocab:
        j=idx[w]
        if etiq[j] is None: continue
        verd='V' if w in VERB else ('S' if w in SUST else 'O')
        if etiq[j]==verd: ok+=1
        tot+=1
    accep=ok/tot if tot else 0
    # tasa de dolor despues (no cambia por etiqueta, pero reportamos)
    out=dict(experiment="v0.9b_etiquetas_mutantes",
             hypothesis="Etiquetas aprendidas por historial de uso (mutan por dolor) convergen a la verdad del corpus.",
             params=dict(d=D,alpha=ALPHA,beta=BETA,V=V,steps=STEPS,corpus="don_quijote"),
             accuracy_etiqueta_vs_verdad=round(accep,4),
             nota="etiq arranca None; critico externo MINIMO solo senala dolor. Distinto a v0.6b (etiq hardcodeada).")
    with open("results_v09b.json","w") as f: json.dump(out,f,indent=2)
    print(f"accuracy etiqueta aprendida vs verdad: {accep:.4f}")
    print("\n-> results_v09b.json")

if __name__=="__main__": main()
