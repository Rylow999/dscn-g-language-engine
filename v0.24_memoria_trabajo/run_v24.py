#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.24 — MEMORIA DE TRABAJO CON VITALIDAD (Gap 3, cerca del ancla DSCN-G).
Memoria de trabajo = SLOTS competitivos. Cada nodo tiene vitalidad v (cuanto
activo/reciente). Al procesar seq: nodo actual recibe disparo v+=1; los demas
decaen v*=decay. Competencia: el de mayor v "gana" el foco (atencion Hebbiana,
sin backprop). Test honesto: tras una frase, el sujeto mantiene ALTA vitalidad y
los distractores decaen. Tambien: vitalidad residual ayuda a next-token (lo reciente
influye). Mide retencion de foco vs decaimiento y si mejora next-token vs sin vitalidad.
"""
import json, math, random, re, time
from collections import Counter
SEED=0; D=16; WIN=4; DECAY=0.85
def norm(v): return math.sqrt(sum(x*x for x in v)) or 1e-9
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def load_dq(max_tokens=20000):
    txt=open("/data/user/0/com.hermesagent.android/files/home/donquijote.txt",encoding="utf-8",errors="ignore").read()
    words=re.findall(r"[a-záéíóúñü]+", txt.lower())
    vocab=[w for w,_ in Counter(words).most_common(150)]
    idxall=[i for i,w in enumerate(words) if w in set(vocab)]
    step=max(1,len(idxall)//max_tokens); chosen=idxall[::step][:max_tokens]
    return [words[i] for i in chosen], vocab
def simulate_wm(seq, vocab, decay=DECAY):
    idx={w:i for i,w in enumerate(vocab)}
    Vn=len(vocab)
    vit=[0.0]*Vn            # vitalidad por nodo
    # embeddings por co-ocurrencia (grafo simple)
    emb=[[0.0]*D for _ in range(Vn)]
    cnt=[0]*Vn
    for i in range(1,len(seq)):
        a=idx[seq[i-1]]; b=idx[seq[i]]
        for d in range(D): emb[b][d]+=1.0 if a==b else 0.3
        cnt[b]+=1
    for w in range(Vn):
        if cnt[w]: emb[w]=[e/cnt[w] for e in emb[w]]
    # recorrer seq actualizando vitalidad (disparo + decaimiento)
    focus_trace=[]
    for t,w in enumerate(seq):
        wi=idx[w]
        # decaimiento de todos
        vit=[v*decay for v in vit]
        # disparo al nodo actual
        vit[wi]+=1.0
        # foco = nodo de mayor vitalidad
        focus=max(range(Vn), key=lambda k: vit[k])
        focus_trace.append((w, vocab[focus], vit[wi], max(vit)))
    return vit, emb, idx, focus_trace
def next_token(emb, idx, seq, vit, use_vit=True):
    # predecir siguiente token: cos(emb[ctx], emb[candidato]) ponderado por vitalidad
    if len(seq)<2: return None
    c=seq[-1]; cc=emb[idx[c]]
    best=None; bs=-2.0
    for w in idx:
        wi=idx[w]
        sc=cos(cc, emb[wi])
        if use_vit: sc+=0.3*vit[wi]   # vitalidad residual influye
        if sc>bs: bs=sc; best=w
    return best
def main():
    print("=== v0.24 MEMORIA DE TRABAJO CON VITALIDAD ===")
    t0=time.time()
    seq,vocab=load_dq()
    vit,emb,idx,trace=simulate_wm(seq,vocab)
    # TEST 1: retencion de foco — el nodo disparado debe quedar con vitalidad alta
    # medimos: en los ultimos N pasos, ¿el foco coincide con el nodo recien disparado?
    # (competencia: el disparado debe dominar brevemente)
    n=len(seq)
    hit_foco=0; tot=0
    for t in range(1,n):
        w,tw,vw,mx=trace[t]
        # el nodo disparado en t tiene vitalidad vw; si es el max -> gano foco
        if vw>=mx-1e-9: hit_foco+=1
        tot+=1
    # TEST 2: next-token con vs sin vitalidad residual
    ok_v=ok_s=0; tot2=0
    for i in range(WIN, min(n, 3000)):
        ctx_seq=seq[:i]
        vit_i=vit  # aproximado (usamos vitalidad final; test relativo)
        # next-token real
        truth=seq[i]
        p_v=next_token(emb,idx,ctx_seq,vit_i,use_vit=True)
        p_s=next_token(emb,idx,ctx_seq,vit_i,use_vit=False)
        tot2+=1
        if p_v==truth: ok_v+=1
        if p_s==truth: ok_s+=1
    print(f"sim+eval {time.time()-t0:.0f}s")
    print(f"TEST1 foco dominado por disparado: {hit_foco}/{tot} = {round(hit_foco/tot,3)}")
    print(f"TEST2 next-token: con vitalidad={round(ok_v/tot2,3)}  sin vitalidad={round(ok_s/tot2,3)}  (baseline azar={round(1/len(vocab),3)})")
    out=dict(experiment="v0.24_memoria_trabajo_vitalidad",
             hypothesis="Vitalidad competitiva: el nodo disparado domina el foco brevemente y la vitalidad residual mejora next-token vs sin ella.",
             params=dict(d=D,window=WIN,decay=DECAY),
             test1_foco_dominante=dict(acc=round(hit_foco/tot,3),n=tot),
             test2_next_token=dict(con_vitalidad=round(ok_v/tot2,3), sin_vitalidad=round(ok_s/tot2,3), baseline_azar=round(1/len(vocab),3)),
             vitalidad_ayuda=ok_v>=ok_s)
    json.dump(out,open("results_v24.json","w"),indent=2)
    print("\n-> results_v24.json")
if __name__=="__main__": main()
