import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
from scipy.linalg import eig
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

def liouvillian_matrix(g, kappa, gamma, gamma_star, delta):
    L = np.array([
        [-gamma,    0,    0,   -2*g],
        [0,    -kappa,    0,   +2*g],
        [0,         0,  -(gamma+gamma_star+kappa)/2, -delta],
        [g,        -g,  +delta, -(gamma+gamma_star+kappa)/2]
    ], dtype=float)
    return L

def compute_indistinguishability(g, kappa, gamma, gamma_star, delta=0):
    L = liouvillian_matrix(g, kappa, gamma, gamma_star, delta)
    lam, V = eig(L)
    V_inv = np.linalg.inv(V)
    c = V_inv @ np.array([1.0, 0.0, 0.0, 0.0])
    alpha_cc = c * V[1, :]
    alpha_ec = c * (V[2, :] + 1j * V[3, :])
    M = np.array([
        [-(gamma+gamma_star)/2.0, -1j * g],
        [-1j*g, -1j * delta - kappa / 2.0]
    ], dtype=complex)
    mu, U = eig(M)
    U_inv = np.linalg.inv(U)
    beta_cc = U[1, :] * U_inv[:, 1]
    beta_ce = U[1, :] * U_inv[:, 0]
    P = np.outer(alpha_cc, beta_cc) + np.outer(alpha_ec, beta_ce)
    T_lam = 1.0 / (-(lam[:, None] + lam.conj()[None, :]))
    T_mu  = 1.0 / (-(mu[:, None]  + mu.conj()[None, :]))
    Q = P @ T_mu @ P.conj().T
    N = np.real(np.sum(Q * T_lam))
    T_lam_D = 1.0 / (-(lam[:, None] + lam[None, :]))
    D = np.real(np.sum(np.outer(alpha_cc, alpha_cc) * T_lam_D * (1.0/(-lam))[None, :]))
    I = N / D if abs(D) > 1e-200 else 0.0
    return I

def compute_product(g, kappa, gamma, gamma_star, delta=0):
    beta = compute_beta(g, kappa, gamma, gamma_star)
    I = compute_indistinguishability(g, kappa, gamma, gamma_star, delta)
    return beta * I

# parameters
gamma = 1e0
gamma_star = 2090 * gamma 
grid_size = 90

# grid
g_vals = np.logspace(-2, 6, grid_size)
k_vals = np.logspace(-2, 6, grid_size)
beta_array = np.zeros((len(k_vals), len(g_vals)))
I_array = np.zeros((len(k_vals), len(g_vals)))
product_array = np.zeros((len(k_vals), len(g_vals)))
for i, kappa in enumerate(k_vals):
    for j, g in enumerate(g_vals):
        b = compute_beta(g * gamma, kappa * gamma, gamma, gamma_star)
        I = compute_indistinguishability(g * gamma, kappa * gamma, gamma, gamma_star)
        beta_array[i, j]    = b
        I_array[i, j]       = I
        product_array[i, j] = b * I
Kx, Gy = np.meshgrid(k_vals, g_vals)
product_plot = product_array.T 

#plot
fig, ax = plt.subplots(figsize=(10, 8))
vmin = product_plot[product_plot > 0].min()
vmax = 1e0
mesh = ax.pcolormesh(Kx, Gy, product_plot, cmap='jet',norm=LogNorm(vmin=vmin, vmax=vmax), shading='auto')
plt.colorbar(mesh, ax=ax, label=r'Funneling ratio', ticks=LogLocator(base=10))
# contour_levels = [] # comment or remove for no contours
# plot_utils.add_labeled_contours(ax, product_plot, contour_levels, Kx, Gy)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e-2, 1e6)
ax.set_ylim(1e-2, 1e6)
ax.set_xlabel(r'$\kappa/\gamma$', fontsize=12)
ax.set_ylabel(r'$g/\gamma$', fontsize=12)
ax.grid(True, alpha=0.2, linestyle=':')
plot_utils.add_literature_points(ax, "B") # comment to not show
plt.tight_layout()
plt.show()
