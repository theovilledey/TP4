import numpy as np
import matplotlib.pyplot as plt
import qutip
import scipy
import itertools
from joblib import Parallel, delayed

# returns (t, beta_t): time grid and cumulative beta up to t_max, with nt time steps
def beta_function(gamma,g,kappa,nt,t_max=None): 
    gamma_star = 1e4*gamma
    # time grid
    if t_max is None:
        t_max = 100/min(g,kappa)
    t = np.linspace(0,t_max,nt)

    # for basis we want <a.dag*a> number operator on cavity so it's projector on |g1>
    # g0 necessary for leakages etc. 
    g0 = qutip.basis(3,0)
    e0 = qutip.basis(3,1)
    g1 = qutip.basis(3,2)
    P_e = e0*e0.dag()
    P_cav1 = g1*g1.dag() # = a.dag*a

    H = g*(e0*g1.dag()+g1*e0.dag())

    c_dissip = [np.sqrt(gamma)*(g0*e0.dag()),np.sqrt(kappa)*(g0*g1.dag()),np.sqrt(gamma_star)*P_e]
    # sqrt from Lindbladform giving rates as |C_i|^2

    rho_0 = P_e
    res = qutip.mesolve(H,rho_0,t,c_dissip,e_ops=[P_cav1],options={'method':'bdf','nsteps':2000000,'atol':1e-6,'rtol':1e-4,'store_states':False})

    rho_cc = np.real(res.expect[0]) # apparently mesolve resolution can make tiny Im parts appear 
    beta_t = kappa*scipy.integrate.cumulative_trapezoid(rho_cc,t,initial=0.0)
    # inital is to make the length the same as t 

    return t,beta_t

# returns first index where remaining rise < tol * total rise, or None if never flat
def relat_converg(beta_t,tol): 
    total = beta_t[-1] - beta_t[0]
    if total == 0:
        return 0
    remaining = (beta_t[-1] - beta_t) / total  # fraction of total rise still to come
    indices = np.where(remaining < tol)[0]
    if len(indices) == 0:
        return None
    return indices[0]

# finds t_max such that beta_t converges well before the end, by doubling from 100/min(g,kappa)
def find_t_max(gamma,g,kappa,tol=0.02,max_doublings=15):
    t_max = 100/gamma  # system drains back to g0 on timescale ~1/gamma regardless of g and kappa
    for _ in range(max_doublings):
        nt_coarse = 200
        t,beta_t = beta_function(gamma,g,kappa,nt=nt_coarse,t_max=t_max)
        idx = relat_converg(beta_t,tol)
        if idx is not None and idx < int(0.9*len(t)):
            return t_max
        t_max *= 10
    raise RuntimeError(f"beta_t did not converge after {max_doublings} doublings (t_max={t_max:.3e})")

# MISSING STEPS
# grid
# make mesh
# compute and plot

# grid computation (beta, tmax, nt)
def compute_grid_point(gamma, g, kappa, nt=200):
    # worker: finds t_max and computes beta_final for one (g, kappa) point
    try:
        t_max = find_t_max(gamma, g, kappa)
        _, beta_t = beta_function(gamma, g, kappa, nt=nt, t_max=t_max)
        return (g, kappa, t_max, nt, beta_t[-1])
    except Exception as e:
        print(f"ERROR at g={g:.2e}, kappa={kappa:.2e}: {e}")
        return (g, kappa, None, None, None)

if __name__ == '__main__':
    gamma_grid = 1
    n_axis     = 7
    kappa_vals = np.logspace(-2, 4, n_axis)
    g_vals     = np.logspace(-2, 4, n_axis)

    args_list = [(gamma_grid, g_i, k_i) for g_i, k_i in itertools.product(g_vals, kappa_vals)]

    print(f"Running {len(args_list)} grid points in parallel")
    results = Parallel(n_jobs=-1, verbose=10, backend='multiprocessing')(delayed(compute_grid_point)(gamma_i, g_i, k_i) for gamma_i, g_i, k_i in args_list)

    # store results in 2D arrays indexed [i_g, i_kappa]
    beta_grid = np.full((n_axis, n_axis), np.nan)
    tmax_grid = np.full((n_axis, n_axis), np.nan)
    nt_grid   = np.full((n_axis, n_axis), np.nan)
    for (g_i, k_i, t_max_i, nt_i, beta_i) in results:
        ig = np.where(g_vals == g_i)[0][0]
        ik = np.where(kappa_vals == k_i)[0][0]
        beta_grid[ig, ik] = beta_i   if beta_i   is not None else np.nan
        tmax_grid[ig, ik] = t_max_i  if t_max_i  is not None else np.nan
        nt_grid[ig, ik]   = nt_i     if nt_i     is not None else np.nan

    print("Done.")
    print("beta_grid:\n", beta_grid)
