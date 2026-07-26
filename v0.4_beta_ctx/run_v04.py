#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G v0.4 — beta_eff contextual (de Pandora: beta*(1+rho)).
rho = densidad de conexiones activas = fraccion de pares con afinidad > umbral.
Pregunta: beta_eff mejora retencion de masa (N*) y convergencia de omega (T2)?
Re-run robusto: N=300, seeds=4, steps=1000 (el sweep N=1000 previo se corto).
"""
import json, math, random, sys, time
ALPHA=5.0; BETA=0.20; GAMMA=0.01; THETA_DEATH=0.10; D=8; K=10; N_INIT=300

def make_omega(rng,scale=0.1): return [rng.gauss(0.0,scale) for _ in range(D)]
def norm(v): return math.sqrt(sum(x*x for x in v))
def dot(a,b): return sum(x*y for x,y in zip(a,b))

class Node:
    __slots__=("omega","vitality","alive")
    def __init__(self,omega): self.omega=omega; self.vitality=1.0; self.alive=True

class Engine:
    def __init__(self,seed,ctx_beta=False):
        self.rng=random.Random(seed); self.omega_ideal=[1.0/math.sqrt(D) for _ in range(D)]
        self.nodes=[Node(make_omega(self.rng)) for _ in range(N_INIT)]
        self.ctx=ctx_beta
        idx=list(range(N_INIT)); self.rng.shuffle(idx); self.chains=idx[:K]
    def _active(self): return [i for i,n in enumerate(self.nodes) if n.alive]
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
    def _rho(self,act):
        if len(act)<2: return 0.0
        c=0
        for a in range(len(act)):
            for b in range(a+1,len(act)):
                d=math.sqrt(sum((x-y)**2 for x,y in zip(self.nodes[act[a]].omega,self.nodes[act[b]].omega)))
                if math.exp(-ALPHA*d)>0.5: c+=1
        pairs=len(act)*(len(act)-1)/2
        return c/pairs if pairs>0 else 0.0
    def step(self):
        act=self._active()
        if not act: return
        activity={i:0.0 for i in act}
        for k in range(K):
            old=self.chains[k]
            if not self.nodes[old].alive: old=self.rng.choice(act); self.chains[k]=old
            new=self._chain_step(old); self.chains[k]=new; activity[new]=activity.get(new,0.0)+1.0
        root=act[0]; activity[root]=activity.get(root,0.0)+1.0
        for i in activity: activity[i]/=(K+1)
        decay=math.exp(-GAMMA)
        for i in act:
            a=activity.get(i,0.0); n=self.nodes[i]
            n.vitality=n.vitality*decay+a*(1.0-decay)
            if n.vitality<THETA_DEATH: n.alive=False
        act=self._active()
        if not act: return
        rho=self._rho(act)
        beta_eff=BETA*(1.0+rho) if self.ctx else BETA
        sel=max(act,key=lambda i:norm(self.nodes[i].omega))
        w=self.nodes[sel].omega; nrm=norm(w)+1e-8; reward=(dot(w,self.omega_ideal)/nrm+1.0)/2.0
        for i in act:
            I=norm(self.nodes[i].omega)
            if I>0:
                be=min(beta_eff,beta_eff*(I/(norm(self.nodes[i].omega)+1e-8)))
                o=self.nodes[i].omega
                self.nodes[i].omega=[(1-be)*o[k]+be*reward*self.omega_ideal[k] for k in range(D)]
    def alive_count(self): return sum(1 for n in self.nodes if n.alive)
    def mean_alignment(self):
        act=self._active()
        if not act: return 0.0
        return sum(dot(self.nodes[i].omega,self.omega_ideal)/(norm(self.nodes[i].omega)+1e-8) for i in act)/len(act)

def run(ctx,steps=1000,seeds=4):
    Ns=[]; Al=[]
    for s in range(seeds):
        e=Engine(seed=s,ctx_beta=ctx)
        for _ in range(steps): e.step()
        Ns.append(e.alive_count()); Al.append(e.mean_alignment())
    return sum(Ns)/len(Ns), sum(Al)/len(Al)

def main():
    print("=== v0.4 beta_eff contextual (K=10, N=300, re-run robusto) ===")
    n0,a0=run(False); n1,a1=run(True)
    print(f"beta fijo    : N*={n0:.1f}  align={a0:.4f}")
    print(f"beta_eff(rho): N*={n1:.1f}  align={a1:.4f}")
    out=dict(experiment="v0.4_beta_ctx",
             note="re-run con N=300,seeds=4,steps=1000 (el sweep N=1000 previo se corto).",
             hypothesis="beta_eff=beta*(1+rho) mejora retencion (N*) y convergencia (align).",
             params=dict(alpha=ALPHA,beta=BETA,gamma=GAMMA,theta_death=THETA_DEATH,d=D,chains=K,N_init=N_INIT),
             results=dict(beta_fixed=dict(N_star=round(n0,2),align=round(a0,4)),
                          beta_ctx=dict(N_star=round(n1,2),align=round(a1,4))))
    with open("results_v04.json","w") as f: json.dump(out,f,indent=2)
    print("\n-> results_v04.json")

if __name__=="__main__": main()
