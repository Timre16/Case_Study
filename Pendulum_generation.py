import numpy as np
import pandas as pd
from scipy.integrate import odeint
import argparse
import json
import os
from multiprocessing import Pool

def simulate(params):
    """Führt eine Doppelpendel-Simulation mit gegebenen Parametern aus."""
    # Unpack parameters
    m1, m2, L1, L2, y0, t, out_dir, idx = params
    def deriv(y, t):
        θ1, ω1, θ2, ω2 = y
        Δ = θ2 - θ1
        denom1 = (m1+m2)*L1 - m2*L1*np.cos(Δ)**2
        denom2 = (L2/L1)*denom1
        dω1 = (m2*L1*ω1**2*np.sin(Δ)*np.cos(Δ) +
               m2*9.81*np.sin(θ2)*np.cos(Δ) +
               m2*L2*ω2**2*np.sin(Δ) -
               (m1+m2)*9.81*np.sin(θ1)) / denom1
        dω2 = (-m2*L2*ω2**2*np.sin(Δ)*np.cos(Δ) +
               (m1+m2)*9.81*np.sin(θ1)*np.cos(Δ) -
               (m1+m2)*L1*ω1**2*np.sin(Δ) -
               (m1+m2)*9.81*np.sin(θ2)) / denom2
        return [ω1, dω1, ω2, dω2]

    sol = odeint(deriv, y0, t)
    df = pd.DataFrame({
        'time': t,
        'theta1': sol[:,0],
        'omega1': sol[:,1],
        'theta2': sol[:,2],
        'omega2': sol[:,3]
    })
    fname = os.path.join(out_dir, f"sim_{idx:04d}.csv")
    df.to_csv(fname, index=False)
    # Speichere Metadaten
    meta = {'m1':m1, 'm2':m2, 'L1':L1, 'L2':L2, 'y0':y0}
    with open(fname.replace('.csv','.json'),'w') as f:
        json.dump(meta, f)
    return fname

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=100, help='Anzahl Simulationen')
    parser.add_argument('--out', type=str, default='data', help='Output-Ordner')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t = np.arange(0, 30, 0.02)

    tasks = []
    for i in range(args.runs):
        # Zufällige Parameter und Anfangszustände
        m1, m2 = np.random.uniform(0.5,2.0,2)
        L1, L2 = np.random.uniform(0.5,2.0,2)
        y0 = [np.pi/2 + np.random.randn()*0.01, 0,
              np.pi/2 + np.random.randn()*0.01, 0]
        tasks.append((m1, m2, L1, L2, y0, t, args.out, i))

    # Parallel ausführen
    with Pool() as pool:
        results = pool.map(simulate, tasks)
    print("Erzeugte Dateien:", results)

if __name__ == '__main__':
    main()
