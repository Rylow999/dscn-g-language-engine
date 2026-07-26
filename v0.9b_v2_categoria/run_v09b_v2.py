#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.9b v2 — CATEGORIZACION con VOCAB BALANCEADO (corrige test inutil).
El v0.9b limpio usaba top-150 donde 93% son sustantivos -> azar=0.9267, el test
no media nada. Aca: vocab = 75 SUST + 75 VERB mas frecuentes (top-400 pool),
balance 50/50. Baseline de azar = 0.50. Tras next-token limpio, clusterizo omega
(k=2) y mido PUREZA. Si > 0.55, la geometria separa S/V sola (baseline 0.50).
"""
import json, math, random, re, time
from collections import Counter
D=16; TOP=400; NSV=75; BETA=0.10; EPOCHS=3; SEED=0; K=2
SUST={'don','quijote','caballero','sancho','casa','gato','pez','perro','rojo','libro','dia','noche','rey','campo','espada','mujer','hombre','agua','pan','vino','tierra','cielo','sol','luna','mano','vida','muerte','amor','mundo','pueblo','castillo','senor','camino','cuerpo','historia','caballo','pueblo','ventana','puerta','sangre','fuego','agua','tierra','monte','valle','rio','fuente','arbol','hoja','flor','fruto','mesa','silla','muro','piedra','hierro','plata','oro','cruz','iglesia','monje','abad','duque','conde','principe','rey','reina','soldado','ejercito','guerra','paz','enemigo','amigo','hijo','padre','madre','hermano','esposa','doncella','criado','labrador','ganado','oveja','cordero','lobo','oso','leon','tigre','serpiente','ave','pez','insecto','flor','hoja','raiz','tronco','ramo','espina','fuego','llama','ceniza','humo','polvo','viento','lluvia','nieve','hielo','frio','calor','luz','sombra'}
VERB={'come','corre','es','tiene','va','dice','hace','da','ve','oye','sabe','quiere','puede','debe','vive','habla','piensa','llega','pone','deja','mira','siente','ama','odia','cree','llama','anda','camina','corre','vuela','duerme','despierta','siente','grita','llora','rie','canta','baila','come','bebe','toma','suena','abre','cierra','entra','sale','vuelve','queda','cae','levanta','sostiene','lleva','trae','envia','escribe','lee','aprende','ensenya','olvida','recuerda','espera','temee','confia','duda','niega','afirma','pregunta','responde','calla','grita','pelea','vence','pierde','muere','nace','crece','engorda','adelgaza','enferma','sana','curra','trabaja','descansa','juega','gana','pierde','paga','compra','vende','roba','guarda','esconde','busca','encuentra','pierde','alcanza','toca','hiere','sana','mata','salva','ayuda','sirve','manda','obedece','reina','gobierna','juzga','condena','perdona','castiga'}
def tok(t): return re.findall(r"[a-záéíóúñü]+", t.lower())
def cos(a,b):
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def avg_vecs(vecs):
    n=len(vecs)
    if n==0: return [0.0]*D
    return [sum(v[d] for v in vecs)/n for d in range(D)]
def kmeans(points,rng,iters=10):
    if len(points)<K: return [0]*len(points)
    cents=[points[rng.randrange(len(points))] for _ in range(K)]
    for _ in range(iters):
        cl=[[] for _ in range(K)]
        for p in points:
            best=max(range(K), key=lambda c: cos(p,cents[c])); cl[best].append(p)
        for k in range(K):
            if cl[k]: cents[k]=avg_vecs(cl[k])
    return [max(range(K), key=lambda c: cos(p,cents[c])) for p in points]
def main():
    print("=== v0.9b v2 CATEGORIZACION (vocab balanceado 75S+75V) ===")
    rng=random.Random(SEED)
    text=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=tok(text)
    pool=[w for w,_ in Counter(words).most_common(TOP)]
    sust=[w for w in pool if w in SUST][:NSV]
    verb=[w for w in pool if w in VERB][:NSV]
    vocab=sust+verb
    V=len(vocab)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(V)]
    seq=[w for w in words if w in set(vocab)]
    idx={w:i for i,w in enumerate(vocab)}
    t0=time.time()
    for ep in range(EPOCHS):
        for i in range(1,len(seq)):
            a,b=idx[seq[i-1]],idx[seq[i]]
            omega[a]=[(1-BETA)*omega[a][k]+BETA*omega[b][k] for k in range(D)]
    print(f"train {time.time()-t0:.0f}s  V={V}")
    pts=[omega[i] for i in range(V)]
    assign=kmeans(pts,rng)
    purity=0
    for k in range(K):
        members=[vocab[i] for i in range(V) if assign[i]==k]
        if not members: continue
        s=sum(1 for w in members if w in SUST)
        v=sum(1 for w in members if w in VERB)
        purity+=max(s,v)
    purity/=V
    out=dict(experiment="v0.9b_v2_categorizacion_balanceada",
             hypothesis="Con vocab 50/50 S-V, la geometria omega separa sintaxis sola (pureza > 0.50 azar).",
             params=dict(d=D,top=TOP,nsv=NSV,beta=BETA,epochs=EPOCHS,k=K,V=V),
             pureza_cluster=round(purity,4), baseline_azar=0.50,
             separa=(purity>0.55))
    json.dump(out,open("results_v09b_v2.json","w"),indent=2)
    print(f"pureza={purity:.4f} (azar 0.50)")
    print("\n-> results_v09b_v2.json")
if __name__=="__main__": main()
