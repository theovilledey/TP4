import numpy as np
import matplotlib.pyplot as plt
import plot_utils

def compute_beta(g, kappa, gamma, gamma_star):
    M = np.array([
        [-gamma,      0,       -2*g],
        [0,       -kappa,       2*g],
        [g,           -g, -(gamma + kappa + gamma_star) / 2.0]
    ], dtype=float)
    x0 = np.array([1.0, 0.0, 0.0])
    X = np.linalg.solve(M, -x0)
    return kappa * X[1]

# parameters (note I put the ratios)
gamma = 1e0
gamma_star = 12502 *gamma
grid_size = 900
SHOW_LITERATURE = False # plots the system?
EMITTER = "A" # which systems to plot?

# grid (be careful put the ratios here not just the values)
g_vals = np.logspace(-2,6,grid_size)
k_vals = np.logspace(-2,6,grid_size)
beta_array = np.zeros((len(k_vals), len(g_vals)))
for i, kappa in enumerate(k_vals):
    for j, g in enumerate(g_vals):
        beta_array[i, j] = compute_beta(g * gamma, kappa * gamma, gamma, gamma_star)

# plotting — kappa on x-axis, g on y-axis
Kx, Gy = np.meshgrid(k_vals, g_vals)
beta_plot = beta_array.T  # beta_plot[i,j] = beta(g=g_vals[i], kappa=k_vals[j])
fig, ax = plt.subplots(figsize=(10, 8))
mesh = ax.pcolormesh(Kx, Gy, beta_plot, vmin=0, vmax=1, cmap='jet', shading='auto')
plt.colorbar(mesh, ax=ax, label='Beta factor β')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('κ/γ', fontsize=12)
ax.set_ylabel('g/γ', fontsize=12)
ax.grid(True, alpha=0.2, linestyle=':')
levels = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999]
plot_utils.add_labeled_contours(ax, beta_plot, levels, Kx, Gy)
if SHOW_LITERATURE:
    plot_utils.add_literature_points(ax, EMITTER)
plt.tight_layout()
plt.show()
