#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, math, random, sys, time
exec(open("/data/user/0/com.hermesagent.android/files/home/run_v01.py").read().split("def main()")[0])

def main():
    plan=[(4,20,2000),(10,20,2000),(50,20,2000),(200,20,2000),(1000,8,1200),(5000,3,800)]
    ub=1.0/THETA_DEATH; rows=[]
    print(f"{'N_init':>8} | {'N* mean':>9} | {'N* std':>7} | {'rho':>6} | fp? | bound?")
    for Ni,se,st in plan:
        t0=time.time(); Ns,rhos=run_N(Ni,se,st)
        Nm=sum(Ns)/len(Ns); Nsd=math.sqrt(sum((x-Nm)**2 for x in Ns)/len(Ns)); rm=sum(rhos)/len(rhos)
        fp=rm>=Nm*THETA_DEATH**2; ubok=Nm<=ub
        rows.append(dict(N_init=Ni,seeds=se,steps=st,N_star_mean=round(Nm,3),N_star_std=round(Nsd,3),
                         rho_mean=round(rm,4),fixed_point_ok=fp,universal_bound_ok=ubok,bound=ub))
        print(f"{Ni:>8} | {Nm:>9.2f} | {Nsd:>7.2f} | {rm:>6.3f} | {'✓' if fp else '✗'}   | {'✓' if ubok else '✗'}   ({time.time()-t0:.0f}s)")
        sys.stdout.flush()
    out=dict(experiment="v0.1_concept_proof",hypothesis="N* escala sublinealmente con N_init (memoria escasa)",
             params=dict(alpha=ALPHA,beta=BETA,gamma=GAMMA,theta_death=THETA_DEATH,d=D,chains=N_CHAINS),
             universal_bound=ub,note="Eq.2+Eq.5. Kuramoto omitido. Hasta N_init=5000 (Python puro).",results=rows)
    with open("results_v01.json","w") as f: json.dump(out,f,indent=2)
    print("\n-> results_v01.json")

if __name__=="__main__": main()
