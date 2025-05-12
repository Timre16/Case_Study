import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation
import matplotlib.patches as patches
import pandas as pd

# Parameter des Doppelpendels
m1, m2 = 1.0, 1.0   # Massen
L1, L2 = 1.0, 1.0   # Längen
g = 9.81            # Erdbeschleunigung

# Bewegungsgleichungen
def deriv(y, t, m1, m2, L1, L2, g):
    θ1, ω1, θ2, ω2 = y
    Δ = θ2 - θ1
    denom1 = (m1 + m2) * L1 - m2 * L1 * np.cos(Δ)**2
    denom2 = (L2 / L1) * denom1

    dω1 = (m2 * L1 * ω1**2 * np.sin(Δ) * np.cos(Δ) +
           m2 * g * np.sin(θ2) * np.cos(Δ) +
           m2 * L2 * ω2**2 * np.sin(Δ) -
           (m1 + m2) * g * np.sin(θ1)) / denom1

    dω2 = (-m2 * L2 * ω2**2 * np.sin(Δ) * np.cos(Δ) +
           (m1 + m2) * g * np.sin(θ1) * np.cos(Δ) -
           (m1 + m2) * L1 * ω1**2 * np.sin(Δ) -
           (m1 + m2) * g * np.sin(θ2)) / denom2

    return [ω1, dω1, ω2, dω2]

# Zeitarray für ~30 Sekunden Simulation
t_max = 30
dt = 0.02
t = np.arange(0, t_max, dt)

# Anfangsbedingungen: [θ1, ω1, θ2, ω2]
y0 = [np.pi/2, 0, np.pi/2 + 0.01, 0]

# Numerische Integration der Bewegungsgleichungen
sol = odeint(deriv, y0, t, args=(m1, m2, L1, L2, g))
θ1 = sol[:, 0]
ω1 = sol[:, 1]
θ2 = sol[:, 2]
ω2 = sol[:, 3]

# Umrechnung in kartesische Koordinaten
x1 = L1 * np.sin(θ1)
y1 = -L1 * np.cos(θ1)
x2 = x1 + L2 * np.sin(θ2)
y2 = y1 - L2 * np.cos(θ2)

# Speichern der Simulationsdaten in einer CSV-Datei
data = pd.DataFrame({
    'time': t,
    'theta1': θ1,
    'omega1': ω1,
    'theta2': θ2,
    'omega2': ω2
})
data.to_csv('double_pendulum_data.csv', index=False)

# Visualisierung der Winkel über die Zeit
plt.figure(figsize=(8, 4))
plt.plot(t, θ1, label='θ1 (rad)')
plt.plot(t, θ2, label='θ2 (rad)')
plt.xlabel('Zeit (s)')
plt.ylabel('Winkel (rad)')
plt.title('Doppelpendel-Winkel über die Zeit')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Animation der Pendelbewegung
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-L1 - L2 - 0.2, L1 + L2 + 0.2)
ax.set_ylim(-L1 - L2 - 0.2, L1 + L2 + 0.2)
ax.set_aspect('equal')
ax.axis('off')  # Achsen ausblenden

# Raumhintergrund
room = patches.Rectangle((-2.0, -2.0), 4.0, 4.0, linewidth=2, edgecolor='black', facecolor='#f0f0f0')
ax.add_patch(room)

# Pendelstangen und Massen
line, = ax.plot([], [], 'o-', lw=2, color='darkblue')

def init():
    line.set_data([], [])
    return (line,)

def update(i):
    this_x = [0, x1[i], x2[i]]
    this_y = [0, y1[i], y2[i]]
    line.set_data(this_x, this_y)
    return (line,)

ani = animation.FuncAnimation(
    fig, update, frames=len(t),
    init_func=init, blit=True, interval=dt*1000
)

plt.title('Doppelpendel-Simulation')
plt.show()
