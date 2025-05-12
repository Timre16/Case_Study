import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import os

# === HIER ANPASSEN ===
CSV_PATH = "C:/Users/timei/Desktop/Python Skripte/Case_Study/data/sim_0010.csv"
L1 = 1.0   # Länge der ersten Stange
L2 = 1.0   # Länge der zweiten Stange
INTERVAL_MS = 20  # Frame-Intervall in ms
ROOM_SIZE = 2.2   # Halber Raum in Einheiten
# =====================

def load_data(csv_file):
    """Lädt die CSV mit Spalten: time, theta1, omega1, theta2, omega2."""
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    return pd.read_csv(csv_file)

def compute_positions(df, L1, L2):
    """Berechnet die Cartesian-Koordinaten der beiden Pendelmasse."""
    theta1 = df['theta1'].values
    theta2 = df['theta2'].values
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2

def animate_pendulum(x1, y1, x2, y2, interval, room_size):
    """Erstellt eine 2D-Frontansicht-Animation im Raum."""
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-room_size, room_size)
    ax.set_ylim(-room_size, room_size)
    ax.set_aspect('equal')
    ax.axis('off')

    # Raumhintergrund
    room = patches.Rectangle(
        (-room_size, -room_size),
        2*room_size, 2*room_size,
        linewidth=2, edgecolor='black', facecolor='#f0f0f0'
    )
    ax.add_patch(room)

    # Linie für die Pendelstangen
    line, = ax.plot([], [], 'o-', lw=2, color='darkblue')

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        xs = [0, x1[frame], x2[frame]]
        ys = [0, y1[frame], y2[frame]]
        line.set_data(xs, ys)
        return (line,)

    ani = animation.FuncAnimation(
        fig, update, frames=len(x1),
        init_func=init, blit=True, interval=interval
    )
    plt.show()

if __name__ == "__main__":
    # Daten laden
    df = load_data(CSV_PATH)
    # Positionen berechnen
    x1, y1, x2, y2 = compute_positions(df, L1, L2)
    # Animation starten
    animate_pendulum(x1, y1, x2, y2, INTERVAL_MS, ROOM_SIZE)
