#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.21 v8e — COMPETENCIA + REPULSION CONDICIONAL.
v0.21 v8d (competencia pura) FALLO: acc_gt=0.471, poli_sep=0/6 (azar). La competencia
sola deja un subnodo estancado y sin repulsion los subnodos colapsan al promedio.
v0.21 v8e combina: (1) COMPETENCIA (solo subnodo k* ganador se actualiza hacia el
vecino), + (2) REPULSION CONDICIONAL (se empuja el subnodo NO ganador hacia el
contexto opuesto, SOLO si hay evidencia de contexto diverso: los 2 subnodos ya
tienen embeddings distintos, cos<theta). La repulsion no es ciega (como v8) ni
incondicional: se aplica al subnodo que la competencia NO actualizó, empujandolo
hacia el contexto actual (que es el 'otro sentido' para el subnodo no ganador).
Instrumento correcto: corpus sintetico CON ground truth + monosemicas + curva.
Hipotesis: si competencia+repulsion condicional da acc_gt>0.9 y poli_sep=6/6,
el grafo rustico separa sentidos. Si da 0.50, el problema es D=16/topologia, no
el mecanismo.
"""
import json, math, random
D=16; W=8; EPOCHS=30; K=2; BETA=0.20; BETA_REP=0.15; ALPHA=0.0; SEED=0; THETA=0.3
def norm(v): return math.sqrt(sum(x*x for x in v))
def cos(a,b):
    na=norm(a); nb=norm(b)
    return 0.0 if na<1e-9 or nb<1e-9 else sum(x*y for x,y in zip(a,b))/(na*nb)
def build_corpus(seed=SEED, n_per_sense=80):
    rng=random.Random(seed)
    poly={
      "banco":  (["dinero","pagar","cuenta","oro","plata","billete","banquero"],
                 ["rio","agua","pez","orilla","puente","corriente","rioja"]),
      "llave":  (["puerta","cerradura","abrir","candado","cerradura","llaves"],
                 ["musica","nota","tono","cancion","melodia","acorde","partitura"]),
      "mouse":  (["computadora","click","pantalla","cable","teclado","raton"],
                 ["animal","cola","hueco","roedor","rato","conejo","campo"]),
      "capital":(["dinero","inversion","bolsa","accion","finanza","mercado"],
                 ["ciudad","pais","provincia","madrid","lima","tierra"]),
      "firma":  (["empresa","contrato","sociedad","negocio","compania"],
                 ["manuscrito","firma","rubrica","documento","dedo","tinta"]),
      "oro":    (["metal","plata","joya","precioso","lingote","moneda"],
                 ["sol","amarillo","brillo","rayo","luz","dorado"]),
    }
    mono={
      "quijote":  ["libro","historia","caballero","mancha","ingenioso","novela"],
      "sancho":   ["escudero","panza","rocinante","insular","gordo","sencillez"],
      "caballero":["andante","espada","honor","armas","valor","caballeria"],
    }
    seq=[]; meta=[]
    def add_block(tokens, sense):
        for t in tokens: seq.append(t); meta.append(sense)
    for w,(sa,sb) in poly.items():
        for _ in range(n_per_sense):
            blk=list(sa[:5])+[w]+list(sa[5:7]); add_block(blk,"A")
            blk=list(sb[:5])+[w]+list(sb[5:7]); add_block(blk,"B")
    for w,cm in mono.items():
        for _ in range(n_per_sense):
            blk=list(cm[:5])+[w]+list(cm[5:7]); add_block(blk,"M")
    return seq, list(dict.fromkeys(seq)), list(poly.keys()), list(mono.keys()), meta
def train_compet_rep(seq, vocab, poly_words, mono_words, meta):
    """COMPETENCIA + REPULSION CONDICIONAL.
    Para 'a' en pos i: contexto = vecinos antes de a. k* = subnodo que mejor
    matchea contexto (COMPETENCIA) -> se actualiza hacia b[0]. El subnodo j=1-k*
    (NO ganador) se empuja AWAY del contexto (REPULSION), SOLO si los 2 subnodos
    ya son distintos (cos<theta) -> evita repulsion ciega sobre monosemicas."""
    Vn=len(vocab); idx={w:i for i,w in enumerate(vocab)}
    rng=random.Random(SEED)
    frac=[[[rng.gauss(0,1) for _ in range(D)] for _ in range(K)] for _ in range(Vn)]
    curve=[]
    for ep in range(EPOCHS):
        for i in range(1,len(seq)):
            a=idx[seq[i-1]]; b=idx[seq[i]]
            cw=list(range(max(0,i-1-W),i-1))
            if not cw: continue
            ctx=[0.0]*D
            for c in cw:
                o=frac[idx[seq[c]]][0]
                for d in range(D): ctx[d]+=o[d]
            ctx=[x/len(cw) for x in ctx]
            # COMPETENCIA: k* = subnodo de a que mejor matchea contexto
            bestk,bestc=-1,-1e9
            for k in range(K):
                c=cos(frac[a][k],ctx)
                if c>bestc: bestc=c; bestk=k
            k=bestk; j=1-k
            # actualizar k* hacia b[0] (competencia)
            newk=[(1-BETA)*frac[a][k][d]+BETA*frac[b][0][d] for d in range(D)]
            frac[a][k]=newk
            # REPULSION CONDICIONAL sobre j (no ganador): empujarlo AWAY del ctx
            nj=norm(frac[a][j])
            if nj>1e-9:
                cjj=cos(frac[a][k], frac[a][j])
                if cjj < THETA:  # solo si ya hay divergencia (contexto diverso)
                    newj=[frac[a][j][d]-BETA_REP*(ctx[d]/nj) for d in range(D)]
                    frac[a][j]=newj
        acc,ps,ms=eval_full(seq,vocab,idx,frac,meta,poly_words,mono_words)
        curve.append((ep,round(acc,3),ps,ms))
        if ep%5==0: print(f"  ep{ep:2} acc_gt={acc:.3f} poli_sep={ps}/{len(poly_words)} mono_sep={ms}/{len(mono_words)}")
    return frac, idx, curve
def eval_full(seq, vocab, idx, frac, meta, poly_words, mono_words):
    correctos=0; total=0; poli_sep=0; mono_sep=0
    for w in poly_words:
        if w not in idx: continue
        wi=idx[w]; occ=[i for i,x in enumerate(seq) if x==w]
        if len(occ)<10: continue
        grupos={}
        for i in occ:
            cw=list(range(max(0,i-W),i))
            if not cw: continue
            ctx=[0.0]*D
            for c in cw:
                o=frac[idx[seq[c]]][0]
                for d in range(D): ctx[d]+=o[d]
            ctx=[x/len(cw) for x in ctx]
            bestk,bestc=-1,-1e9
            for k in range(K):
                c=cos(frac[wi][k],ctx)
                if c>bestc: bestc=c; bestk=k
            grupos.setdefault(bestk,0); grupos[bestk]+=1
        n=len(occ); dom=max(grupos.values()) if grupos else n
        if len(grupos)>=2 and dom<n*0.85: poli_sep+=1
    for w in mono_words:
        if w not in idx: continue
        wi=idx[w]; occ=[i for i,x in enumerate(seq) if x==w]
        if len(occ)<10: continue
        grupos={}
        for i in occ:
            cw=list(range(max(0,i-W),i))
            if not cw: continue
            ctx=[0.0]*D
            for c in cw:
                o=frac[idx[seq[c]]][0]
                for d in range(D): ctx[d]+=o[d]
            ctx=[x/len(cw) for x in ctx]
            bestk,bestc=-1,-1e9
            for k in range(K):
                c=cos(frac[wi][k],ctx)
                if c>bestc: bestc=c; bestk=k
            grupos.setdefault(bestk,0); grupos[bestk]+=1
        n=len(occ); dom=max(grupos.values()) if grupos else n
        if len(grupos)>=2 and dom<n*0.85: mono_sep+=1
    for i in range(1,len(seq)):
        w=seq[i]; sense=meta[i]
        if sense not in ("A","B") or w not in idx: continue
        wi=idx[w]; cw=list(range(max(0,i-W),i))
        if not cw: continue
        ctx=[0.0]*D
        for c in cw:
            o=frac[idx[seq[c]]][0]
            for d in range(D): ctx[d]+=o[d]
        ctx=[x/len(cw) for x in ctx]
        bestk,bestc=-1,-1e9
        for k in range(K):
            c=cos(frac[wi][k],ctx)
            if c>bestc: bestc=c; bestk=k
        esperado=0 if sense=="A" else 1
        if bestk==esperado: correctos+=1
        total+=1
    return (correctos/total if total else 0.0), poli_sep, mono_sep
def main():
    print("=== v0.21 v8e COMPETENCIA + REPULSION CONDICIONAL ===")
    seq,vocab,poly_words,mono_words,meta=build_corpus()
    print(f"vocab={len(vocab)} poly={len(poly_words)} mono={len(mono_words)} seq={len(seq)}")
    frac,idx,curve=train_compet_rep(seq,vocab,poly_words,mono_words,meta)
    acc,ps,ms=eval_full(seq,vocab,idx,frac,meta,poly_words,mono_words)
    print(f"FINAL: acc_gt={acc:.3f} poli_sep={ps}/{len(poly_words)} mono_sep={ms}/{len(mono_words)}")
    print(f"curva: {curve}")
    out=dict(experiment="v0.21_v8e_competencia_mas_repulsion_condicional",
             hypothesis="Competencia (k* ganador) + repulsion condicional (empuja subnodo no ganador AWAY del ctx, solo si cos<theta) separa polisemicas. acc_gt>0.9 => viable.",
             params=dict(d=D,w=W,beta=BETA,beta_rep=BETA_REP,alpha=ALPHA,theta=THETA,epochs=EPOCHS),
             resultados=dict(acc_gt=round(acc,3),poli_sep=ps,mono_sep=ms,curve=curve))
    json.dump(out,open("results_v21_v8e.json","w"),indent=2)
    print("-> results_v21_v8e.json")
if __name__=="__main__": main()