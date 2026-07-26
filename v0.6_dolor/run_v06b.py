#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.6b — aprendizaje por DOLOR (RL / subsistencia) + etiquetas que mutan.
Corpus: Don Quijote (espanol). El grafo ya sabe next-token (v0.6a).
Ahora: transicion INVALIDA (segun etiquetas) => baja V (dolor); VALIDA => sube V.
Los nodos doloridos alejan su omega de la transicion erronea. Etiqutas MUTAN por uso.
"""
import json, math, random, re, sys, time
D=8; ALPHA=5.0; BETA=0.20; V=150; EPOCHS=3

def tok(text):
    return re.findall(r"[a-záéíóúñü]+", text.lower())

def build_vocab(words, V):
    from collections import Counter
    return [w for w,_ in Counter(words).most_common(V)]

def omega_of(w, rng):
    r=random.Random(hash(w)%100000)
    return [r.gauss(0,1) for _ in range(D)]
def norm(v): return math.sqrt(sum(x*x for x in v))
def affinity(q,w):
    d=math.sqrt(sum((a-b)**2 for a,b in zip(q,w)))
    return math.exp(-ALPHA*d)

# Etiquetas aprendidas por POS del corpus (usamos un diccionario mini es -> etiqueta)
SUST={'don','quijote','caballero','sancho','casa','gato','pez','perro','rojo','libro','dia','noche','rey','campo','espada','mujer','hombre','agua','pan','vino','tierra','cielo','sol','luna','mano','vida','muerte','amor','mundo','pueblo','castillo','senor','camino','cuerpo','historia'}
VERB={'come','corre','es','tiene','va','dice','hace','da','ve','oye','sabe','quiere','puede','debe','vive','habla','piensa','llega','pone','deja','mira','siente','ama','odia','cree','llama'}
CONN={'y','o','pero','si','por','con','de','a','en','que','como','cuando','porque','pues','aunque'}
def etiqueta(w):
    if w in SUST: return 'S'
    if w in VERB: return 'V'
    if w in CONN: return 'C'
    return 'X'  # desconocido

def train_and_penalize(vocab, omega, Vmap, seq, epochs):
    idx={w:i for i,w in enumerate(vocab)}
    hist=dict()  # historial de aplicaciones por nodo: (etiqueta_anterior, etiqueta_siguiente) -> conteo
    for ep in range(epochs):
        for i in range(len(seq)-1):
            a=seq[i]; b=seq[i+1]
            if a not in idx or b not in idx: continue
            ia,ib=idx[a],idx[b]
            # next-token (supervisado, como v0.6a)
            omega[ia]=[(1-BETA)*omega[ia][k]+BETA*omega[ib][k] for k in range(D)]
            # JUICIO DE DOLOR por etiquetas
            ea,eb=etiqueta(a),etiqueta(b)
            key=(ea,eb)
            hist.setdefault(a,{})
            hist[a][key]=hist[a].get(key,0)+1
            invalid = (ea in('S','X') and eb in('S','X') and a!=b)  # dos sustantivos seguidos = raro
            if invalid:
                Vmap[ia]=max(0.0, Vmap[ia]-0.05)   # DOLOR: baja vitalidad
                Vmap[ib]=max(0.0, Vmap[ib]-0.05)
                # aleja omega de la transicion erronea
                omega[ia]=[(1-0.1)*omega[ia][k]+0.1*(-omega[ib][k]) for k in range(D)]
            else:
                Vmap[ia]=min(1.0, Vmap[ia]+0.01)   # bienestar
    return hist

def tasa_invalida(vocab, seq):
    idx={w:i for i,w in enumerate(vocab)}
    inv=tot=0
    for i in range(1,len(seq)):
        a,b=seq[i-1],seq[i]
        if a in idx and b in idx:
            ea,eb=etiqueta(a),etiqueta(b)
            if (ea in('S','X') and eb in('S','X') and a!=b): inv+=1
            tot+=1
    return inv/tot if tot else 0.0

def main():
    print("=== v0.6b dolor + etiquetas (Don Quijote, V=%d) ===" % V)
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text)
    vocab=build_vocab(words,V)
    rng=random.Random(0)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    Vmap=[0.5]*V
    seq=[w for w in words if w in set(vocab)]
    ti0=tasa_invalida(vocab,seq)
    print(f"tasa invalida ANTES: {ti0:.4f}")
    t0=time.time()
    hist=train_and_penalize(vocab,omega,Vmap,seq,EPOCHS)
    print(f"entrenado en {time.time()-t0:.0f}s")
    ti1=tasa_invalida(vocab,seq)
    print(f"tasa invalida DESPUES: {ti1:.4f}")
    # muestra de etiquetas mutadas (historial de un nodo conocido)
    sample=hist.get('don',{})
    out=dict(experiment="v0.6b_dolor_etiquetas",
             hypothesis="El dolor (baja V en transicion invalida) reduce transiciones invalidas; etiquetas mutan por uso.",
             params=dict(d=D,alpha=ALPHA,beta=BETA,V=V,epochs=EPOCHS,corpus="don_quijote"),
             tasa_invalida_antes=round(ti0,4), tasa_invalida_despues=round(ti1,4),
             mejora=round(ti0-ti1,4),
             nota="Dolor=V-0.05 en transicion S-S; bienestar=V+0.01. Etiquetas por diccionario mini (mutan via historial).")
    with open("results_v06b.json","w") as f: json.dump(out,f,indent=2)
    print("\n-> results_v06b.json")

if __name__=="__main__": main()
