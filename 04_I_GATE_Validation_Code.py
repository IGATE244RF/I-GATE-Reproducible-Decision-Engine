"""
I-GATE Major Revision Reproducibility Script
Manuscript: Intelligent Optimization of Gate Decisions in Complex Product Development Using I-GATE Dynamic Game
Seed: 42
Experiment: 100 Monte Carlo runs x 8 periods x 3 scenarios
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_RUNS = 100
HORIZON = 8
GRID = np.array([0.0, 0.25, 0.50, 0.75, 1.0])

def clip(x, lo, hi):
    return np.clip(x, lo, hi)

def transition(x, u, w):
    C,Q,R,T,K,M = x
    uT,uS,uM,uA = u
    wm,ws,wt = w
    Cn = C + 0.55*uT + 0.35*uS + 0.25*uA + 0.10*uM + 0.15 + 0.08*abs(wm)
    Qn = clip(Q + 0.10*uT + 0.03*uM + 0.02*K - 0.03*abs(wt), 0, 1)
    Rn = clip(R - 0.09*uT - 0.11*uS - 0.04*uA - 0.03*K + 0.05*abs(ws), 0.02, 1)
    Tn = T + 0.30*uT + 0.18*uS + 0.12*uA + 0.10*uM + 0.12
    Kn = clip(K + 0.08*uT + 0.10*uM, 0, 1)
    Mn = clip(M + 0.10*uM - 0.03*abs(wm), 0, 1)
    return np.array([Cn,Qn,Rn,Tn,Kn,Mn])

def lifecycle_value(x):
    C,Q,R,T,K,M = x
    return 120 - (C + 2.5*R + 0.7*T) + 5*Q + 3*K + 4*M

def traditional_policy(x, t, w):
    return np.array([0.25,0.15,0.15,0.10])

def system_dynamics_policy(x, t, w):
    C,Q,R,T,K,M = x
    return np.array([
        clip(0.25 + 0.45*(0.55-Q),0,1),
        clip(0.15 + 0.55*R,0,1),
        clip(0.10 + 0.45*(0.60-M),0,1),
        clip(0.10 + 0.35*R,0,1)
    ])

# Five stakeholder agents: Engineering, Supplier, Manufacturing,
# Management, Customer/Market.
GAMMA = np.array([0.4,0.6,0.4,0.5,0.3])
COST = np.array([0.2,0.2,0.25,0.2,0.15])

def target_vector(x):
    C,Q,R,T,K,M = x
    return clip(np.array([
        0.5 + 0.5*(0.6-Q),
        0.4 + 0.5*R,
        0.4 + 0.3*(0.6-Q),
        0.4 + 0.3*R + 0.2*(0.5-M),
        0.3 + 0.4*(0.6-Q) + 0.2*(0.5-M)
    ]),0,1)

def nash_reference(x, tol=1e-9, max_iter=100):
    b = target_vector(x)
    u = b.copy()
    for _ in range(max_iter):
        old = u.copy()
        for i in range(5):
            others = np.delete(u,i)
            u[i] = (b[i] + GAMMA[i]*others.mean())/(1+GAMMA[i]+COST[i])
        if np.max(np.abs(u-old)) < tol:
            break
    return b,u

def igate_policy(x, t, w):
    _, uN = nash_reference(x)
    reference = np.array([uN[0],uN[1],uN[2],uN[3]])
    market_reference = uN[4]
    best_obj = np.inf
    best_u = None
    for uT in GRID:
        for uS in GRID:
            for uM in GRID:
                for uA in GRID:
                    u = np.array([uT,uS,uM,uA])
                    x1 = transition(x,u,w)
                    x2 = transition(x1,u,np.zeros(3))
                    C,Q,R,T,K,M = x2
                    obj = (C + 7*R + 0.45*T - 3*Q - 1.5*K - 1.5*M
                           + 0.04*np.sum(u*u)
                           + 0.8*np.sum((u-reference)**2)
                           + 0.8*(uM-market_reference)**2)
                    if obj < best_obj:
                        best_obj = obj
                        best_u = u.copy()
    return best_u

def run_policy(policy, shocks):
    x = np.array([1.0,0.45,0.65,1.0,0.25,0.45])
    states=[x.copy()]
    controls=[]
    for t,w in enumerate(shocks):
        u=policy(x,t,w)
        controls.append(u)
        x=transition(x,u,w)
        states.append(x.copy())
    return np.array(states),np.array(controls)

def main(output_dir="outputs"):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED)
    scenarios=[
        ("Traditional Stage-Gate",traditional_policy),
        ("System Dynamics",system_dynamics_policy),
        ("I-GATE",igate_policy)
    ]
    final_rows=[]; control_rows=[]; shock_rows=[]
    for run in range(N_RUNS):
        shocks=rng.normal(0,1,(HORIZON,3))*np.array([0.18,0.15,0.12])
        for t,w in enumerate(shocks):
            shock_rows.append([run,t,*w])
        for name,policy in scenarios:
            states,controls=run_policy(policy,shocks)
            C,Q,R,T,K,M=states[-1]
            LV=lifecycle_value(states[-1])
            final_rows.append([run,name,C,Q,R,T,K,M,LV])
            for t,u in enumerate(controls):
                control_rows.append([run,name,t,*u])
    finals=pd.DataFrame(final_rows,columns=["run","scenario","C","Q","R","T","K","M","Lifecycle_Value"])
    controls=pd.DataFrame(control_rows,columns=["run","scenario","period","u_technology","u_supplier","u_market","u_architecture"])
    shocks=pd.DataFrame(shock_rows,columns=["run","period","market_shock","supplier_shock","technology_shock"])
    finals.to_csv(out/"validation_final_states.csv",index=False)
    controls.to_csv(out/"validation_controls.csv",index=False)
    shocks.to_csv(out/"validation_shocks.csv",index=False)
    summary=finals.groupby("scenario").mean(numeric_only=True)
    trad=summary.loc["Traditional Stage-Gate"]
    ig=summary.loc["I-GATE"]
    lvi=(ig["Lifecycle_Value"]-trad["Lifecycle_Value"])/trad["Lifecycle_Value"]
    rr=(trad["R"]-ig["R"])/trad["R"]
    print(summary)
    print("LVI:",lvi)
    print("Risk reduction:",rr)

if __name__=="__main__":
    main()
