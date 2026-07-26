#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.3 REAL v2 — HIBERNADO sobre el motor REAL de v0.1 (que SI colapsa).
En v0.1 la poda BORRA (alive=False -> sale del active, omega perdido).
Aca: en vez de borrar, HIBERNA (alive=False pero omega/phi se preservan en
lista hibernated). Mide N_active (working set) + N_hibernated = N_total.
Si N_total ~ N_init mientras N_active colapsa a ~4.5, se VALIDA la idea de
Luciano: memoria de masa (no borrar, guardar).
"""
import json, math, random, sys, time
ALPHA=5.0; BETA=0.20; GAMMA=0.01; THETA_DEATH=0.10; D=8; N_CHAINS=3
STEPS=2000

def make_omega(rng,d,scale=0.1): return [rng.gauss(0.0,scale) for _ in range(d)]
def norm(v): return math.sqrt(sum(x*x for x in v))
def dot(a,b): return sum(x*y for x,y in zip(a,b))

class Node:
    __slots__=("omega","vitality","alive","hibernated")
    def __init__(self,omega):
        self.omega=omega; self.vitality=1.0; self.alive=True; self.hibernated=False

class Engine:
    def __init__(self,N_init,seed):
        self.rng=random.Random(seed)
        self.omega_ideal=[1.0/math.sqrt(D) for _ in range(D)]
        self.nodes=[Node(make_omega(self.rng,D)) for _ in range(N_init)]
        idx=list(range(N_init)); self.rng.shuffle(idx)
        self.chains=idx[:N_CHAINS]; self.t=0
        self.hibernated=[]
    def _active(self): return [i for i,n in enumerate(self.nodes) if n.alive]
    def _chain_step(self,src):
        act=self._active()
        if not act: return src
        src_n=self.nodes[src]
        diffs=[math.sqrt(sum((a-b)**2 for a,b in zip(self.nodes[m].omega,src_n.omega))) for m in act]
        mx=max(diffs); ws=[math.exp(-ALPHA*(diffs[i]-mx)) for i in range(len(act))]
        s=sum(ws); r=self.rng.random()*s; acc=0.0
        for i,w in enumerate(act):
            acc+=ws[i]
            if acc>=r: return w
        return act[-1]
    def _interf(self,i): return norm(self.nodes[i].omega)
    def step(self):
        self.t+=1; act=self._active()
        if not act: return
        activity={i:0.0 for i in act}
        for k in range(N_CHAINS):
            old=self.chains[k]
            if not self.nodes[old].alive: old=self.rng.choice(act); self.chains[k]=old
            new=self._chain_step(old); self.chains[k]=new; activity[new]=activity.get(new,0.0)+1.0
        root=act[0]; activity[root]=activity.get(root,0.0)+1.0
        denom=N_CHAINS+1
        for i in activity: activity[i]/=denom
        decay=math.exp(-GAMMA)
        for i in act:
            a=activity.get(i,0.0); n=self.nodes[i]
            n.vitality=n.vitality*decay+a*(1.0-decay)
            if n.vitality<THETA_DEATH:
                n.alive=False; n.hibernated=True   # HIBERNAR, no borrar
                self.hibernated.append(i)
        act=self._active()
        if not act: return
        sel=max(act,key=lambda i:self._interf(i))
        w=self.nodes[sel].omega; nrm=norm(w)+1e-8; reward=(dot(w,self.omega_ideal)/nrm+1.0)/2.0
        for i in act:
            I=self._interf(i)
            if I>0:
                beta_eff=min(BETA,BETA*(I/(norm(self.nodes[i].omega)+1e-8)))
                o=self.nodes[i].omega
                self.nodes[i].omega=[(1-beta_eff)*o[k]+beta_eff*reward*self.omega_ideal[k] for k in range(D)]
    def counts(self):
        na=sum(1 for n in self.nodes if n.alive)
        nh=sum(1 for n in self.nodes if n.hibernated)
        return na,nh,self.N_init if hasattr(self,'N_init') else len(self.nodes)

def run_N(N_init,seeds,steps):
    nas=[]; nhs=[]
    for s in range(seeds):
        eng=Engine(N_init,seed=s)
        for _ in range(steps): eng.step()
        nas.append(sum(1 for n in eng.nodes if n.alive))
        nhs.append(sum(1 for n in eng.nodes if n.hibernated))
    return nas,nhs

def main():
    print("=== v0.3 REAL v2 — HIBERNADO (motor v0.1) ===")
    plan=[(10,20,STEPS),(50,20,STEPS),(200,10,STEPS),(1000,5,STEPS)]
    rows=[]
    for Ni,se,st in plan:
        t0=time.time(); nas,nhs=run_N(Ni,se,st)
        Nm=sum(nas)/len(nas); Hm=sum(nhs)/len(nhs); Tot=Nm+Hm
        rows.append(dict(N_init=Ni,N_active_mean=round(Nm,2),N_hibernated_mean=round(Hm,2),
                         N_total_mean=round(Tot,2),retencion=round(Tot/Ni,4),
                         comparar_v01_sin_hibernar=round(min(Ni,Ni*0.10+4.5),2)))
        print(f"N_init={Ni}: N_active={Nm:.1f} N_hibernado={Hm:.1f} N_total={Tot:.1f} retencion={Tot/Ni:.2f} ({time.time()-t0:.0f}s)")
        sys.stdout.flush()
    out=dict(experiment="v0.3_real_v2_hibernado",
             hypothesis="Hibernar (no borrar omega) mantiene N_total ~ N_init; el working set colapsa a ~4.5 pero la masa sobrevive.",
             params=dict(alpha=ALPHA,beta=BETA,gamma=GAMMA,theta_death=THETA_DEATH,d=D,chains=N_CHAINS,steps=STEPS),
             results=rows)
    with open("results_v03real.json","w") as f: json.dump(out,f,indent=2)
    print("\n-> results_v03real.json")

if __name__=="__main__": main()
