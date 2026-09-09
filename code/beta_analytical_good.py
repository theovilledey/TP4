import numpy as np
import qutip
import matplotlib.pyplot as plt
import plot_utils

# analytically solve by inverting the louiville superoperator (drho/dt= L rho)
# see https://arxiv.org/html/2412.13661v1 for method
gamma = 1.0
gamma_star = 1e4 * gamma

# define the base of the open system as before
g0 = qutip.basis(3, 0) # ground (excitation nowhere)
e0 = qutip.basis(3, 1) # emitter excited, cavity empty
g1 = qutip.basis(3, 2) # emitter relaxed, photon in cavity
# the operators here are for probability - not applied
P_e = e0 * e0.dag() # projector on excited emitter
P_cav = g1 * g1.dag() # projector on cavity
# we need to have the Lindblad collapse operators
# (spontaneous emission - cavity decay - dephasing)
# only dephasing is fixed; emission and cavity decay scale with gamma and kappa
L_emiss = np.sqrt(gamma) * (g0 * e0.dag())
L_deph  = np.sqrt(gamma_star) * P_e
# initialise system
rho_0  = P_e
# steady state that the system will reach eventually
rho_ss = g0 * g0.dag()

# matrices to numpy vectors (flatten makes them (9,) for .conj later)
rho0_vec   = qutip.operator_to_vector(rho_0).full().flatten() # rho_0 = vectorized initial state
rho_ss_vec = qutip.operator_to_vector(rho_ss).full().flatten() # rho_ss = vectorized steady state
P_vec = qutip.operator_to_vector(P_cav).full().flatten() # vectorized projector P_cav

# solve Lx = -(rho_0 - rho_ss) and then beta = kappa * Re(<P|x>)
rhs = -(rho0_vec - rho_ss_vec)

# precompute matrix parts that are fixed (independent of g and kappa)
zero_H = qutip.Qobj(np.zeros((3, 3)))
L_H_unit    = qutip.liouvillian(e0 * g1.dag() + g1 * e0.dag(), []).full()
L_emiss_mat = qutip.liouvillian(zero_H, [L_emiss]).full()
L_deph_mat  = qutip.liouvillian(zero_H, [L_deph]).full()
# L_kappa scales as kappa (since sqrt(kappa)^2 = kappa in the Lindblad superoperator)
L_kappa_unit = qutip.liouvillian(zero_H, [g0 * g1.dag()]).full() # scales with kappa

def beta_analytical(g, kappa):
    L_mat = g * L_H_unit + L_emiss_mat + kappa * L_kappa_unit + L_deph_mat
    x, _, _, _ = np.linalg.lstsq(L_mat, rhs, rcond=None)
    return kappa * np.real(P_vec.conj() @ x) # should check for imaginary part for consistency?

# grid
vals = np.logspace(-2, 6, 900)
beta_grid = np.zeros((len(vals), len(vals)))
for i, g in enumerate(vals):
    for j, kappa in enumerate(vals):
        beta_grid[i, j] = beta_analytical(g * gamma, kappa * gamma)

# plot of beta — kappa on x-axis, g on y-axis (beta_grid[i,j] = beta(g=vals[i], kappa=vals[j]))
Kx, Gy = np.meshgrid(vals, vals)
fig, ax = plt.subplots(figsize=(10, 8))
mesh = ax.pcolormesh(Kx, Gy, beta_grid, vmin=0, vmax=1, cmap='jet', shading='auto')
plt.colorbar(mesh, ax=ax, label=r'$\beta$')
ax.set_xscale('log')
ax.set_yscale('log')
levels = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999]
plot_utils.add_labeled_contours(ax, beta_grid, levels, Kx, Gy)
ax.set_xlabel(r'$\kappa / \gamma$')
ax.set_ylabel(r'$g / \gamma$')

plt.tight_layout()
plt.show()
