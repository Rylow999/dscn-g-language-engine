#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.9c — SUBSISTENCIA GLOBAL (dolor EMERGENTE, no critico externo).
El grafo tiene vitalidad GLOBAL G (0..1). Cada paso:
- G baja por "desajuste interno" (senal de dolor propia): si la afinidad media
  del working set cae (el grafo esta "confundido"), G baja. Eso ES el dolor:
  el sistema se daña a si mismo por su propia dinamica, no por un critico.
- El aprendizaje apunta a MANTENER G alto (active inference rustico, Friston):
  cuando G baja, los omega se ajustan para subir la afinidad media.
- Si G<=0, el grafo "muere" (reinicia). Medimos: ¿G sobrevive mas con el
  ajuste (aprendizaje por subsistencia) que sin el? Eso prueba que el dolor
  INTERNO obliga al sistema a cambiar para evitar lo que lo produce.
"""
import json, math, random, sys, time
ALPHA=5.0; BETA=0.20; GAMMA=0.01; THETA_DEATH=0.10; D=8; N_CHAINS=3; N_INIT=200
STEPS=3000

def make_omega(rng,scale=0.1): return [rng.gauss(0.0,scale) for _ in range(D)]
def norm(v): return math.sqrt(sum(x*x for x in v))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cos(a,b):
    na=norm(a); nb=norm(b)
    if na<1e-9 or nb<1e-9: return 0.0
    return dot(a,b)/(na*nb)

class Node:
    __slots__=("omega","vitality","alive")
    def __init__(self,omega):
        self.omega=omega; self.vitality=1.0; self.alive=True

class Engine:
    def __init__(self,seed,learn=True):
        self.rng=random.Random(seed)
        self.omega_ideal=[1.0/math.sqrt(D) for _ in range(D)]
        self.nodes=[Node(make_omega(self.rng)) for _ in range(N_INIT)]
        idx=list(range(N_INIT)); self.rng.shuffle(idx); self.chains=idx[:N_CHAINS]
        self.G=1.0; self.learn=learn; self.deaths=0
    def _active(self): return [i for i,n in enumerate(self.nodes) if n.alive]
    def _aff_mean(self,act):
        if len(act)<2: return 1.0
        s=0; c=0
        for a in range(len(act)):
            for b in range(a+1,len(act)):
                s+=cos(self.nodes[act[a]].omega,self.nodes[act[b]].omega); c+=1
        return s/c if c else 1.0
    def _chain_step(self,src):
        act=self._active()
        if not act: return src
        sn=self.nodes[src].omega
        diffs=[math.sqrt(sum((a-b)**2 for a,b in zip(self.nodes[m].omega,sn))) for m in act]
        mx=max(diffs); ws=[math.exp(-ALPHA*(diffs[i]-mx)) for i in range(len(act))]
        s=sum(ws); r=self.rng.random()*s; acc=0.0
        for i,w in enumerate(act):
            acc+=ws[i]
            if acc>=r: return w
        return act[-1]
    def step(self):
        act=self._active()
        if not act:
            self.G=max(0,self.G-0.1); self.deaths+=1
            if self.G<=0:
                # muere: reinicia nodos
                for n in self.nodes:
                    n.omega=make_omega(self.rng); n.vitality=1.0; n.alive=True
                self.G=1.0
            return
        # afinidad media = senal de "bienestar"
        aff=self._aff_mean(act)
        # DOLOR EMERGENTE: si afinidad baja, G baja (el sistema se dania)
        if aff < 0.5:
            self.G=max(0,self.G-(0.5-aff)*0.1)
        else:
            self.G=min(1.0,self.G+0.02)
        # actividad
        activity={i:0.0 for i in act}
        for k in range(N_CHAINS):
            old=self.chains[k]
            if not self.nodes[old].alive: old=self.rng.choice(act); self.chains[k]=old
            new=self._chain_step(old); self.chains[k]=new; activity[new]=activity.get(new,0.0)+1.0
        root=act[0]; activity[root]=activity.get(root,0.0)+1.0
        for i in activity: activity[i]/=(N_CHAINS+1)
        decay=math.exp(-GAMMA)
        for i in act:
            a=activity.get(i,0.0); n=self.nodes[i]
            n.vitality=n.vitality*decay+a*(1.0-decay)
            if n.vitality<THETA_DEATH: n.alive=False
        act=self._active()
        if not act: return
        if self.learn and self.G<0.7:
            # aprendizaje por subsistencia: ajusta omega para subir afinidad media
            sel=max(act,key=lambda i:norm(self.nodes[i].omega))
            w=self.nodes[sel].omega; nrm=norm(w)+1e-8; reward=(dot(w,self.omega_ideal)/nrm+1.0)/2.0
            for i in act:
                I=norm(self.nodes[i].omega)
                if I>0:
                    be=min(BETA,BETA*(I/(norm(self.nodes[i].omega)+1e-8)))
                    o=self.nodes[i].omega
                    self.nodes[i].omega=[(1-be)*o[k]+be*reward*self.omega_ideal[k] for k in range(D)]
    def G_final(self): return self.G
    def alive_count(self): return sum(1 for n in self.nodes if n.alive)

def run(learn,seeds=5):
    Gs=[]; As=[]
    for s in range(seeds):
        e=Engine(seed=s,learn=learn)
        for _ in range(STEPS): e.step()
        Gs.append(e.G_final()); As.append(e.alive_count())
    return sum(Gs)/len(Gs), sum(As)/len(As)

def main():
    print("=== v0.9c SUBSISTENCIA GLOBAL (dolor emergente) ===")
    g0,a0=run(False); g1,a1=run(True)
    print(f"SIN aprender por subsistencia: G={g0:.4f}  N*={a0:.1f}")
    print(f"CON aprender por subsistencia: G={g1:.4f}  N*={a1:.1f}")
    out=dict(experiment="v0.9c_subsistencia_global",
             definicion_dolor="senal interna que obliga al sistema a cambiar para evitar lo que lo produce (Luciano)",
             hypothesis="Dolor emergente (baja de G por desajuste) + aprendizaje por subsistencia mantiene G vivo mas tiempo.",
             params=dict(alpha=ALPHA,beta=BETA,gamma=GAMMA,theta_death=THETA_DEATH,d=D,chains=N_CHAINS,N_init=N_INIT,steps=STEPS),
             results=dict(sin_aprender=dict(G=round(g0,4),N_star=round(a0,2)),
                          con_aprender=dict(G=round(g1,4),N_star=round(a1,2))),
             nota="Dolor INTERNO (no critico externo). Si G sobrevive mas con aprender, el dolor obliga a cambiar.")
    with open("results_v09c.json","w") as f: json.dump(out,f,indent=2)
    print("\n-> results_v09c.json")

if __name__=="__main__": main()
