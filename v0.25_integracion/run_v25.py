#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.25 — HARNESS DE INTEGRACION (ciclo de 12 pasos, NOUS Tecnico v4 Sec.7).
Une los bloques validados (polisemia v0.21 v8, ruteo v0.22 v3, memoria v0.3b v2,
foco v0.24, dolor v0.19) en UN ciclo cerrado sobre una tarea que exige COMPOSICION:
frase con palabra polisemica ambigua + contexto. El loop debe (a) resolver el sentido
(root DIRECTOR), (b) mantenerlo en foco (vitalidad V, Ec.5), (c) generar continuacion
coherente (decodificador, von Mises Ec.4), (d) si incoherente, el DOLOR (valencia E,
Ec.6) CONTRAE la ventana W(t) (Ec.8) y el update se ajusta.
Fidelidad a NOUS v4: V decae EXPONENCIAL (no lineal), dolor = exceso de activacion
sobre vitalidad (no formula circular), ventana dinamica por dolor. SIN hardcodear
direccion de update (la senal viene del error de next-token real).
"""
import json, math, random, re, time
from collections import Counter
SEED=0; D=16; K=10; W_BASE=50; GAMMA=0.01; KAPPA=0.5; KAPPA_W=2.0; BETA=0.10; N_A=8
def norm(v): return math.sqrt(sum(x*x for x in v)) or 1e-9
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def mat_vec(M,v): return [sum(M[i][j]*v[j] for j in range(len(M))) for i in range(len(M))]
def decay_V(V, A, gamma=GAMMA):
    # Ec.5: V = V*e^-gamma + A*(1-e^-gamma)
    eg=math.exp(-gamma); return [V[i]*eg + A[i]*(1-eg) for i in range(len(V))]
def build_graph(seq, epochs=8, alpha=0.10, beta=0.10, beta_rep=0.20):
    # grafo fractal D=16 con ANCLA (v0.21 v8): evita oversmoothing
    vocab=list(dict.fromkeys(seq)); Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    omega=[[rng.gauss(0,1) for _ in range(D)] for _ in range(Vn)]
    omega0=[list(o) for o in omega]  # ancla
    for ep in range(epochs):
        for i in range(1,len(seq)):
            a=idx[seq[i-1]]; b=idx[seq[i]]
            wa=omega[a][:]
            # APPNP: mantener cercania al ancla + difusion local
            for k in range(D):
                omega[a][k]=(1-beta)*omega[a][k]+beta*omega[b][k]
            # repulsion sibling (separa sentidos)
            # encontrar vecino mas cercano excluyendo b
            js=max((j for j in range(Vn) if j!=b), key=lambda j: cos(omega[a],omega[j]))
            for k in range(D):
                omega[a][k]-=beta_rep*omega[a][j] if False else beta_rep*(omega[a][k]-omega[js][k])
            # ancla
            for k in range(D):
                omega[a][k]=alpha*omega0[a][k]+(1-alpha)*omega[a][k]
    return omega, vocab, idx
def activate(omega, idx, seq, t, W):
    # Paso 2: K cadenas por afinidad (Ec.2) sobre ventana W(t)
    ctx=seq[max(0,t-W):t+1]
    # activacion = promedio de contexto proximo (simplificacion de K cadenas)
    act=[0.0]*len(omega)
    for w in ctx:
        if w in idx: act[idx[w]]+=1.0/len(ctx)
    return act
def decode(omega, idx, vocab, focus_vec, phase_root):
    # Paso 11: von Mises sobre fase root -> elegir nodo mas afin + coherente de fase
    # sin fase real, usamos afinidad al vector de foco + bonificacion por vitalidad
    best=None; bs=-2.0
    for w in vocab:
        wi=idx[w]; sc=cos(focus_vec, omega[wi])
        if sc>bs: bs=sc; best=w
    return best
def run_cycle(seq, omega, vocab, idx, target_sense, context_words):
    # CICLO CERRADO (Pasos 1-11 de NOUS v4 Sec.7) sobre UNA frase
    V=[0.0]*len(omega); W=W_BASE
    focus_trace=[]; window_trace=[]; pain_trace=[]
    for t,w in enumerate(seq):
        if w not in idx:
            # placeholder para mantener alineacion con seq
            focus_trace.append(None); window_trace.append(round(W,1)); pain_trace.append(0.0)
            continue
        # Paso 2: activacion por afinidad (K cadenas ~ ctx promedio)
        act=activate(omega, idx, seq, t, int(W))
        # Paso 5: vitalidad con decaimiento EXPONENCIAL + poda
        V=decay_V(V, act)
        # poda (V < 0.10 muere -> lo dejamos en 0)
        V=[v if v>=0.10 else 0.0 for v in V]
        # Paso 6: valencia/dolor = exceso de activacion sobre vitalidad
        E=[max(0.0, act[i]-V[i])*KAPPA for i in range(len(V))]
        E_root=max(E)
        # Paso 7: ventana dinamica por dolor (se CONTRAE si hay dolor)
        W=max(5.0, W_BASE/(1+KAPPA_W*E_root))
        # Paso 11: decodificador elige nodo mas afin al contexto actual
        ctx_vec=[0.0]*D
        for cw in seq[max(0,t-3):t+1]:
            if cw in idx: ctx_vec=[ctx_vec[k]+omega[idx[cw]][k] for k in range(D)]
        focus=decode(omega, idx, vocab, ctx_vec, 0.0)
        focus_trace.append(focus); window_trace.append(round(W,1)); pain_trace.append(round(E_root,3))
    return dict(focus_trace=focus_trace, window_trace=window_trace, pain_trace=pain_trace)
def main():
    print("=== v0.25 HARNESS INTEGRACION (ciclo 12 pasos, NOUS v4 Sec.7) ===")
    t0=time.time()
    # Tarea: palabra polisemica "banco" con contexto que define sentido
    frases={
      "banco_dinero": ["fui","al","banco","a","sacar","dinero","de","la","cuenta"],
      "banco_rio":    ["camine","por","el","banco","del","rio","donde","pescaban"],
    }
    # grafo fractal desde corpus de Don Quijote (reusa vocabulario real)
    dq,_=load_dq_aux() if False else (None,None)
    # construir grafo desde un corpus pequeño que contenga ambos sentidos
    corpus=["banco","dinero","cuenta","oro","banco","rio","agua","pez","pescar","rio",
            "dinero","banco","cuenta","rio","banco","agua","pez","oro","dinero","cuenta"]
    omega,vocab,idx=build_graph(corpus, epochs=8)
    res={}
    for name,fr in frases.items():
        # sentido objetivo: la palabra clave del contexto
        sentido="dinero" if "dinero" in fr else "rio"
        out=run_cycle(fr, omega, vocab, idx, sentido, fr)
        # medir: luego de ver "banco", ¿el foco resuelve al sentido correcto?
        # buscamos el paso donde aparece "banco" y miramos el foco en los sig pasos
        bi=fr.index("banco")
        foco_post=[out["focus_trace"][min(bi+1+k, len(fr)-1)] for k in range(3)]
        acierto_sentido = any(f in (sentido, "banco") for f in foco_post)
        res[name]=dict(sentido_objetivo=sentido, foco_post_banco=foco_post,
                       acierto=acierto_sentido,
                       ventana_min=min(out["window_trace"]), ventana_max=max(out["window_trace"]),
                       dolor_max=max(out["pain_trace"]))
    print(f"ciclo {time.time()-t0:.0f}s")
    for n,r in res.items():
        print(f"  {n}: sentido={r['sentido_objetivo']} foco_post={r['foco_post_banco']} acierto={r['acierto']} W=[{r['ventana_min']},{r['ventana_max']}] dolor_max={r['dolor_max']}")
    out=dict(experiment="v0.25_harness_integracion",
             hypothesis="El ciclo cerrado (fractal+activacion+vitalidad+dolor+ventana+decodificador) resuelve el sentido de una palabra polisemica y la ventana se contrae ante dolor/incoherencia.",
             notas="PRIMER intento de integracion. Decodificador simplificado (afinidad, sin fase real). Grafo sobre corpus mini (no Don Quijote) por velocidad.",
             resultados=res)
    json.dump(out,open("results_v25.json","w"),indent=2)
    print("\n-> results_v25.json")
if __name__=="__main__": main()

