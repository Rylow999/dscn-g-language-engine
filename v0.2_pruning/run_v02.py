#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DSCN-G Language Engine v0.2 — Poda redesign (parameter sweep).
Pregunta: el colapso de N* en ~4.5 (v0.1) ¿es parametrico o estructural?
Teoria: N*_max ≈ (K+1)/theta_death  (presupuesto de visitas / umbral).
Barremos (K, theta_death) a N_init=1000 fijo y medimos N*.
Luego, al mejor config, testeamos scaling N* vs N_init.
"""
import json, math, random, sys, time

ALPHA=5.0; BETA=0.20; GAMMA=0.01; D=8

def make_omega(rng,scale=0.1): return [rng.gauss(0.0,scale) for _ in range(D)]
def norm(v): return math.sqrt(sum(x*x for x in v))
def dot(a,b): return sum(x*y for x,y in zip(a,b))

class Node:
    __slots__=("omega","vitality","alive")
    def __init__(self,omega): self.omega=omega; self.vitality=1.0; self.alive=True

class Engine:
    def __init__(self,N_init,seed,K,theta_death):
        self.rng=random.Random(seed)
        self.omega_ideal=[1.0/math.sqrt(D) for _ in range(D)]
        self.nodes=[Node(make_omega(self.rng)) for _ in range(N_init)]
        self.K=K; self.theta_death=theta_death
        idx=list(range(N_init)); self.rng.shuffle(idx)
        self.chains=idx[:K]
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
        act=self._active()
        if not act: return
        activity={i:0.0 for i in act}
        for k in range(self.K):
            old=self.chains[k]
            if not self.nodes[old].alive: old=self.rng.choice(act); self.chains[k]=old
            new=self._chain_step(old); self.chains[k]=new; activity[new]=activity.get(new,0.0)+1.0
        root=act[0]; activity[root]=activity.get(root,0.0)+1.0
        denom=self.K+1
        for i in activity: activity[i]/=denom
        decay=math.exp(-GAMMA)
        for i in act:
            a=activity.get(i,0.0); n=self.nodes[i]
            n.vitality=n.vitality*decay+a*(1.0-decay)
            if n.vitality<self.theta_death: n.alive=False
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
    def alive_count(self): return sum(1 for n in self.nodes if n.alive)

def run(N_init,seeds,steps,K,theta_death):
    Ns=[]
    for s in range(seeds):
        e=Engine(N_init,seed=s,K=K,theta_death=theta_death)
        for _ in range(steps): e.step()
        Ns.append(e.alive_count())
    return Ns

def main():
    print("=== v0.2 SWEEP (K, theta_death) a N_init=1000 ===")
    sweep=[(3,0.10),(3,0.03),(3,0.01),(10,0.03),(10,0.01),(30,0.01),(30,0.003)]
    sweep_rows=[]
    for K,th in sweep:
        t0=time.time(); Ns=run(1000,5,1000,K,th); Nm=sum(Ns)/len(Ns); Nsd=math.sqrt(sum((x-Nm)**2 for x in Ns)/len(Ns))
        cap=(K+1)/th
        sweep_rows.append(dict(K=K,theta_death=th,N_init=1000,N_star_mean=round(Nm,2),N_star_std=round(Nsd,2),cap=round(cap,1),frac_alive=round(Nm/1000,3)))
        print(f"K={K:>3} θ={th:<6} -> N*={Nm:>7.1f}±{Nsd:.1f}  (tope {(K+1)/th:.0f})  {time.time()-t0:.0f}s")
        sys.stdout.flush()

    # scaling en el mejor config (mayor N*)
    best=max(sweep_rows,key=lambda r:r["N_star_mean"])
    print(f"\n=== SCALING en mejor config K={best['K']} θ={best['theta_death']} ===")
    scale_rows=[]
    for Ni in [200,1000,5000]:
        t0=time.time(); Ns=run(Ni,3,800,best["K"],best["theta_death"]); Nm=sum(Ns)/len(Ns)
        scale_rows.append(dict(N_init=Ni,K=best["K"],theta_death=best["theta_death"],N_star_mean=round(Nm,2),frac_alive=round(Nm/Ni,3)))
        print(f"N_init={Ni:>5} -> N*={Nm:>8.1f}  ({Nm/Ni*100:.1f}% vivo)  {time.time()-t0:.0f}s")
        sys.stdout.flush()

    out=dict(experiment="v0.2_pruning_redesign",
             hypothesis="El colapso de N* es parametrico: subiendo K / bajando theta_death, N* escala.",
             params=dict(alpha=ALPHA,beta=BETA,gamma=GAMMA,d=D),
             note="Eq.2+Eq.5. Kuramoto omitido. N*_max teorico ≈ (K+1)/theta_death.",
             sweep=sweep_rows, scaling=scale_rows)
    with open("results_v02.json","w") as f: json.dump(out,f,indent=2)
    print("\n-> results_v02.json")

if __name__=="__main__": main()
