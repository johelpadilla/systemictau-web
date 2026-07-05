import numpy as np
from scipy.stats import kendalltau
import pandas as pd

def logistic_map(r, x):
    return r * x * (1 - x)

def get_superstable_orbit(r, k):
    period = 2**k
    x = 0.5
    # transient
    for _ in range(10000):
        x = logistic_map(r, x)
    
    # orbit
    orbit = []
    for _ in range(period):
        x = logistic_map(r, x)
        orbit.append(x)
    return np.array(orbit)

def compute_theorem_24v(k, r_val):
    orbit = get_superstable_orbit(r_val, k)
    # rank permutation
    ranks = np.argsort(np.argsort(orbit))
    
    period = 2**k
    taus = []
    # all cyclic shifts
    for shift in range(period):
        shifted_ranks = np.roll(ranks, shift)
        tau, _ = kendalltau(ranks, shifted_ranks)
        taus.append(np.abs(tau))
        
    e_tau = np.mean(taus)
    theoretical = (2**(k-1)) / (2**k - 1)
    
    return {
        "k": k,
        "Period": period,
        "E[|tau|] Empirical": e_tau,
        "E[|tau|] Theoretical (Teorema24v)": theoretical,
        "Difference": np.abs(e_tau - theoretical)
    }

# Known superstable parameters for the logistic map
r_super = [
    3.2360679775, # k=1
    3.4985616993, # k=2
    3.5546408627, # k=3
    3.5666673798, # k=4
    3.5692435316, # k=5
    3.5697952937, # k=6
    3.5699134654, # k=7
    3.5699387742  # k=8
]

results = []
for k in range(1, 7):
    res = compute_theorem_24v(k, r_super[k-1])
    results.append(res)

df = pd.DataFrame(results)
print("Teorema 24v - Coherencia Ordinal Exacta (Árbol Diádico)")
print("="*60)
print(df.to_string(index=False))
print("="*60)
print(f"Límite empírico hacia k->infinito aprox: {df['E[|tau|] Empirical'].iloc[-1]:.4f} (Teórico: 0.50)")
