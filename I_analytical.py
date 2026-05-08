# see report for a breakdown of the maths involved to understand what is done
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
import warnings
warnings.filterwarnings('ignore')

def liouvillian_matrix(g, kappa, gamma, gamma_star, delta):
    # vectorize the density matrix (rho_ee,rho_cc,Re(rho_ec),Im(rho_ec))
    # this allows testing for different values of gamma etc
    # the reason we take Re() and Im() is just a basis choice to have a real matrix
    L = np.array([
        [-gamma,    0,    0,   -2*g],
        [0,    -kappa,    0,   +2*g],
        [0,         0,  -(gamma+gamma_star+kappa)/2, -delta],
        [g,        -g,  +delta, -(gamma+gamma_star+kappa)/2]
    ], dtype=float)
    return L

def density_matrix_components(t, g, kappa, gamma, gamma_star, delta):
    # to compute rho(t) = e^(Lt)*rho(0) analytically use eigendecomposition
    # build L
    # initial state (1,0,0,0)
    # return result rho_ee, rho_ec, rho_ec
    L = liouvillian_matrix(g, kappa, gamma, gamma_star, delta)
    # Eigenvalue decomposition
    eigenvals, eigenvecs = eig(L)
    rho_0 = np.array([1.0, 0.0, 0.0, 0.0])
    V_inv = np.linalg.inv(eigenvecs) # e^(L*t) = V*e^(Λ*t)*V^(-1)
    rho_t_vec = eigenvecs @ np.diag(np.exp(eigenvals * t)) @ V_inv @ rho_0
    rho_ee = np.real(rho_t_vec[0])
    rho_cc = np.real(rho_t_vec[1])
    rho_ec = rho_t_vec[2] + 1j * rho_t_vec[3]
    return rho_ee, rho_cc, rho_ec

def green_function_time_domain(tau, g, kappa, gamma, gamma_star, delta):
    # compute retarded green as 2x2 matrix
    # used for two time correlator
    # gives amplitude at t+tau
    # enforce causality? (claude)
    if tau <= 0:
        return np.zeros((2, 2), dtype=complex)
    # Propagator matrix M = -iH-Sigma^R (eq. 27)
        # diagonal = decay of each mode 
        # off-diagonal = coupling
    M = np.array([
        [-(gamma + gamma_star) / 2.0, -1j*g],
        [-1j*g, -1j*delta - kappa / 2.0]
    ], dtype=complex)
    eigenvals, eigenvecs = eig(M)
    # G^R(tau) = exp(M * tau) use eigendecomposition
    V_inv = np.linalg.inv(eigenvecs)
    G_R_tau = eigenvecs @ np.diag(np.exp(eigenvals * tau)) @ V_inv
    return G_R_tau

def two_time_correlator_func(t, tau, g, kappa, gamma, gamma_star, delta):
    # compute the correlator as trace Tr(a.dag*G^R(tau)*a*rho(t)) = ⟨a.dag(t+tau)*a(t)⟩
    _,rho_cc,rho_ec = density_matrix_components(t, g, kappa, gamma, gamma_star, delta)
    G_R = green_function_time_domain(tau, g, kappa, gamma, gamma_star, delta)
    # G_R[1,1] = G_cc^R and G_R[1,0] = G_ce^R
    # intuition G_R[1,1] is the cavity-cavity propagation
    # G_R[1,0] emitter-cavity channel feeds at t+tau
    correlator = G_R[1, 1] * rho_cc + G_R[1, 0] * rho_ec
    return correlator

def single_time_intensity_func(t, g, kappa, gamma, gamma_star, delta):
    # ⟨a.dag(t)*a(t)⟩ = rho_cc(t)
    _,rho_cc,_ = density_matrix_components(t, g, kappa, gamma, gamma_star, delta)
    return rho_cc

# analytical by eigendecomposition (eq. 16 supplement) - exact solution
# use closed form ∫exp(xt)dt=-1/x possible as Re(x<0) (express all as complex exp)
def compute_indistinguishability(g, kappa, gamma, gamma_star, delta=0):
    # 1 - Liouvillian eigendecompostition to express rho as sum of exp
    # rho(t) = e^(Lt) rho(0) = V e^(Λt) V^-1 rho(0) = Σ_i e^(λ_i t) |ψ_i><ψ_i| rho(0)
    L = liouvillian_matrix(g, kappa, gamma, gamma_star, delta) 
    lam, V = eig(L) 
    V_inv = np.linalg.inv(V)
    c = V_inv @ np.array([1.0,0.0,0.0,0.0]) # project init state on eigenbasis
    alpha_cc = c*V[1,:] # sum coefficients for cavity population 
    alpha_ec = c * (V[2, :] + 1j * V[3, :]) # sum coefs for coherence
    # G^R(tau)=exp(M*tau) (eq. 27 supplement) - green function eigendecomposition
    M = np.array([
        [-(gamma+gamma_star)/2.0, -1j * g],
        [-1j*g, -1j * delta - kappa / 2.0]
    ], dtype=complex)
    mu, U = eig(M) 
    U_inv = np.linalg.inv(U)
    beta_cc = U[1, :] * U_inv[:, 1] # same thing - coefs for sum
    beta_ce = U[1, :] * U_inv[:, 0] 
    # two-time correlator coefficients 
    P = np.outer(alpha_cc, beta_cc) + np.outer(alpha_ec, beta_ce) # (4,2) array
    # analytically evaluate numerator (product of coefs / sum of eigenvalues)
    T_lam = 1.0 / (-(lam[:, None] + lam.conj()[None,:])) 
    T_mu  = 1.0 / (-(mu[:, None]  + mu.conj()[None, :])) 
    Q = P @ T_mu @ P.conj().T 
    N = np.real(np.sum(Q*T_lam))
    # same thing for denominator (just see report)
    T_lam_D = 1.0 / (-(lam[:, None] + lam[None, :])) 
    D = np.real(np.sum(np.outer(alpha_cc, alpha_cc)*T_lam_D*(1.0/(-lam))[None,:]))
    if abs(D) > 1e-200: # avoid the artifact?
        I = N / D
    else:
        I = 0.0
    return I # min or max this in case of rounding problem

# parameters
gamma = 1e0
gamma_star = 1e2 * gamma
grid_size = 90

# grid
g_vals = np.logspace(-2,6,grid_size)
k_vals = np.logspace(-2,6,grid_size)
G, K = np.meshgrid(g_vals, k_vals)
I_array = np.zeros_like(G)
total = len(g_vals)*len(k_vals) # just for print()
count = 0

# main run
for i, kappa in enumerate(k_vals):
    for j, g in enumerate(g_vals):
        count += 1 # comment next line if no need to see progress
        print(f"{count:2d}/{total}", flush=True)
        try:
            I = compute_indistinguishability(g * gamma, kappa * gamma, gamma, gamma_star)
            I_array[i, j] = I
        except Exception as e:
            print(f"ERROR: {e}")
            I_array[i, j] = 0.0

# plotting
Kx, Gy = np.meshgrid(k_vals, g_vals) # Kx[i,j]=k_vals[j],Gy[i,j]=g_vals[i]
I_plot = I_array.T # I_plot[i,j] = I(g=g_vals[i], κ=k_vals[j])
fig, ax = plt.subplots(figsize=(10, 8))
levels = np.linspace(0, 1, 50)
contour = ax.contourf(Kx, Gy, I_plot,levels=levels, cmap='jet')
cbar = plt.colorbar(contour, ax=ax, label='Indistinguishability I')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e-2, 1e6)
ax.set_ylim(1e-2, 1e6)
ax.set_xlabel('κ/γ', fontsize=12)
ax.set_ylabel('g/γ', fontsize=12)
ax.grid(True, alpha=0.2, linestyle=':')

# plotting litterature values (remove if needed)
k_points = np.array([3275, 320, 1544])
g_points = np.array([45,   5.8, 180])
# [3275, 320, 1544, 12.8, 5.7, 0.57]
# [45,   5.8, 180,  81,   72,  1.3]
labels = ["Device 1","Device 2","Device 3"]
for k, g, lab, in zip(k_points, g_points, labels):
    ax.scatter(k,g,s=80,zorder=5,label=lab)
ax.legend(fontsize=9, loc='best')

plt.tight_layout()
plt.show()