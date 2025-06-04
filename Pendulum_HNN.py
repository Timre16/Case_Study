import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import subprocess
import math
import matplotlib.animation as animation


# === EIGENE EINSTELLUNGEN ===
DATA_FOLDER = "C:/Users/timei/Desktop/Python Skripte/Case_Study/data"
LOG_DIR     = "runs/double_pendulum_hnn"
EPOCHS      = 20
LR          = 1e-3
BATCH_SIZE  = 64
TRAIN_SPLIT = 0.8
# ============================

def load_all_data(folder):
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Kein .csv in '{folder}' gefunden.")
    dfs = [pd.read_csv(f)[['theta1','omega1','theta2','omega2']].values
           for f in csv_files]
    data = np.vstack(dfs)
    return torch.tensor(data, dtype=torch.float32)

class HNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: [batch, 4] = [q1, p1, q2, p2]
        x = x.clone().requires_grad_(True)
        H = self.net(x)
        gradH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        dq1, dp1 = gradH[:,1], -gradH[:,0]
        dq2, dp2 = gradH[:,3], -gradH[:,2]
        return torch.stack([dq1, dp1, dq2, dp2], dim=1)

def train(model, data, writer):
    """
    Trainiert das gegebene HNN-Modell auf den Daten und loggt
    Batch- und Epoche-Metriken in TensorBoard.
    """
    # Hyperparameter
    n = data.size(0)
    train_n = int(n * TRAIN_SPLIT)
    # Shuffle & Split
    perm = torch.randperm(n)
    train_data = data[perm[:train_n]]
    test_data  = data[perm[train_n:]]
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()

    # Initiales Histogram-Logging der Gewichte
    for name, param in model.named_parameters():
        writer.add_histogram(f"Weights/{name}", param, 0)

    # Anzahl Batches pro Epoche (ceil braucht math)
    num_batches = int(math.ceil(train_data.size(0) / BATCH_SIZE))
    global_step = 0

    for epoch in range(1, EPOCHS+1):
        epoch_loss = 0.0
        print(f"Starting epoch {epoch}/{EPOCHS}")

        for i in range(0, train_data.size(0), BATCH_SIZE):
            batch = train_data[i:i+BATCH_SIZE]

            # true_deriv hier ersetzen durch deine Finite-Differences
            # Wenn dein Datensatz nacheinander sortierte Zeitpunkte enthält:
            dt = 0.02  # Beispiel-Zeitschritt
            true_deriv = torch.zeros_like(pred)


            pred = model(batch)
            loss = loss_fn(pred, true_deriv)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            writer.add_scalar("Loss/Batch", loss.item(), global_step)
            global_step += 1

        avg_loss = epoch_loss / num_batches
        test_loss = loss_fn(model(test_data), torch.zeros_like(test_data))

        writer.add_scalar("Loss/Train", avg_loss, epoch)
        writer.add_scalar("Loss/Test", test_loss.item(), epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("Training/Epoch", epoch, epoch)

        print(f"Finished epoch {epoch}/{EPOCHS} — train_loss: {avg_loss:.4e}, test_loss: {test_loss:.4e}")

    print(f"Total training steps: {global_step}")
    torch.save(model.state_dict(), "hnn_model.pth")

    writer.close()


def simulate_hnn(model, y0, t_max=20, dt=0.02):
    times = np.arange(0, t_max, dt)
    y = torch.tensor(y0, dtype=torch.float32).unsqueeze(0)  # shape: [1, 4]
    y.requires_grad_(True)  # wichtig für .grad()

    traj = [y0]

    for _ in times[1:]:
        dy = model(y)  # erwartet y mit shape [1, 4], gibt auch [1, 4] zurück

        # Sicherstellen, dass dy nicht die Graph-Verfolgung fortsetzt
        dy = dy.detach()

        y = y + dy * dt  # Euler-Schritt
        y.retain_grad()  # falls nötig
        traj.append(y.squeeze(0).detach().numpy())

    return np.array(traj)


def plot_trajectory(traj):
    L1, L2 = 1.0, 1.0
    θ1, θ2 = traj[:,0], traj[:,2]
    x1 = L1 * np.sin(θ1); y1 = -L1 * np.cos(θ1)
    x2 = x1 + L2 * np.sin(θ2); y2 = y1 - L2 * np.cos(θ2)

    plt.figure(figsize=(6,6))
    plt.plot(x2, y2, lw=1)
    plt.title("HNN-Predicted Double Pendulum Trajectory")
    plt.axis('equal')
    plt.show()

def animate_double_pendulum(traj, interval=30, save=False):
    L1, L2 = 1.0, 1.0
    θ1, θ2 = traj[:, 0], traj[:, 2]

    x1 = L1 * np.sin(θ1)
    y1 = -L1 * np.cos(θ1)
    x2 = x1 + L2 * np.sin(θ2)
    y2 = y1 - L2 * np.cos(θ2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal')
    ax.set_title("Double Pendulum Simulation (HNN)")

    line, = ax.plot([], [], 'o-', lw=2, color='royalblue')
    trail, = ax.plot([], [], '-', lw=1, color='lightblue', alpha=0.6)
    trail_x, trail_y = [], []

    def init():
        line.set_data([], [])
        trail.set_data([], [])
        return line, trail

    def update(frame):
        trail_x.append(x2[frame])
        trail_y.append(y2[frame])
        line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
        trail.set_data(trail_x, trail_y)
        return line, trail

    ani = animation.FuncAnimation(
        fig, update, frames=len(θ1), init_func=init,
        interval=interval, blit=True, repeat=False
    )

    if save:
        ani.save("double_pendulum_simulation.mp4", fps=30, dpi=200)

    plt.show()

if __name__ == "__main__":
    # 1) TensorBoard starten
    proc = subprocess.Popen([
        "tensorboard",
        f"--logdir={LOG_DIR}",
        "--port=6006"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"TensorBoard läuft (PID {proc.pid}) – öffne http://localhost:6006")

    # 2) Daten laden, Modell und Writer initialisieren
    data   = load_all_data(DATA_FOLDER)
    model  = HNN()
    writer = SummaryWriter(log_dir=LOG_DIR)

    # 3) Training mit Logging
    train(model, data, writer)

    # 4) Simulation & Plot
    traj = simulate_hnn(model, [np.pi/2, 0, np.pi/2 + 0.01, 0])
    plot_trajectory(traj)
    animate_double_pendulum(traj)


    # 5) TensorBoard beenden
    proc.terminate()
    print("TensorBoard beendet.")