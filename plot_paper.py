import numpy as np
import matplotlib.pyplot as plt
import qutip
import scipy
import itertools

def beta_function(gamma,g,kappa,nt,t_max=None): # returns (t, beta_t): time grid and cumulative beta up to t_max, with nt time steps

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
    res = qutip.mesolve(H,rho_0,t,c_dissip,e_ops=[P_cav1],options={'nsteps':50000,'atol':1e-8,'rtol':1e-6,'store_states': False})

    rho_cc = np.real(res.expect[0]) # apparently mesolve resolution can make tiny Im parts appear 
    beta_t = kappa*scipy.integrate.cumulative_trapezoid(rho_cc,t,initial=0.0)
    # inital is to make the length the same as t 

    return t,beta_t

def relat_converg(beta_t,tol): # returns first index where remaining rise < tol * total rise, or None if never flat

    total = beta_t[-1] - beta_t[0]
    if total == 0:
        return 0
    remaining = (beta_t[-1] - beta_t) / total  # fraction of total rise still to come
    indices = np.where(remaining < tol)[0]
    if len(indices) == 0:
        return None
    return indices[0]

def find_t_max(gamma,g,kappa,nt_coarse=200,tol=0.005,max_doublings=15): # finds t_max such that beta_t converges well before the end, by doubling from 100/min(g,kappa)
    t_max = 100/min(g,kappa)
    for _ in range(max_doublings):
        t,beta_t = beta_function(gamma,g,kappa,nt=nt_coarse,t_max=t_max)
        idx = relat_converg(beta_t,tol)
        if idx is not None and idx < int(0.9*len(t)):
            return t_max
        t_max *= 2
    raise RuntimeError(f"beta_t did not converge after {max_doublings} doublings (t_max={t_max:.3e})")

# relative change in final beta value as nt doubles
rel_diff_nt = np.abs(np.diff(beta_finals)) / np.abs(beta_finals[:-1])
print("rel change in beta_final as nt doubles:", rel_diff_nt)

# testing convergence for a single (g, kappa) point
gamma = 1
g     = 10e3
kappa = 10e6
t_max = find_t_max(gamma, g, kappa)
print(f"Converged t_max = {t_max:.3e}")
nt_test = [200, 400, 800, 1600, 3200]
beta_finals = []
for nt in nt_test:
    t, beta_t = beta_function(gamma, g, kappa, nt=nt, t_max=t_max)
    beta_finals.append(beta_t[-1])
    plt.plot(t, beta_t, label=f'nt={nt}')
plt.xlabel('t')
plt.ylabel('beta(t)')
plt.legend()
plt.title(f'nt convergence — g={g:.1e}, kappa={kappa:.1e}, t_max={t_max:.2e}')
plt.show()

# MISSING STEPS: 
# 1 - FIND CONVERGED PARAMETERS NT AND T_MAX FOR (k,g)
# 2 - STORE THEM IN 2D GRIP
# 3 - RUN COMPUTATION OF BETA
# 4 - PLOT BETA

# STEP 1 LOGIC
# compute beta with a small nt until we get a converegnce (flat tail) -> t_max
# use that t_max and increase nt until it doesn't change result  