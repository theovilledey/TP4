import numpy as np
import matplotlib.pyplot as plt
import qutip
import scipy
import itertools

def beta_function(gamma,g,kappa,nt): # nt the number of time steps

    gamma_star = 1e4*gamma
    # time grid
    t_max = 1 # 100/min(g,kappa) 
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

def relat_converg(beta_t,tol):

    rel_diff = np.abs(np.diff(beta_t)) / np.abs(beta_t[:-1])
    idx = np.where(rel_diff<tol)[0][0] + 1

    return idx

# testing values to get a feeling
beta_finals = []
nt_test = np.array([200])
for i in nt_test:
    t,beta_t = beta_function(gamma=1,g=10e3,kappa=10e6,nt=i)
    print("x value:", t[relat_converg(beta_t,0.005)])
    beta_finals.append(beta_t[-1])
    plt.plot(t,beta_t,'.',label=f'nt={i}')
plt.show()



# # running nt convergence
# beta_finals = []
# nt_test = nt_test = np.array([200,400,800,1600,3200,6400])
# for i in nt_test:
#     t,beta_t = beta_function(gamma=1,g=1000,kappa=0.1,nt=i)
#     beta_finals.append(beta_t[-1])
#     plt.plot(t,beta_t,label=f'nt={i}')
# plt.show()

# # showing the progression of precision
# rel_diff = np.abs(np.diff(beta_finals))/np.abs(beta_finals[:-1])
# # mid_nt = np.sqrt(nt_test[1:]*nt_test[:-1])
# plt.plot(nt_test,beta_finals,'+')
# plt.xscale('log')
# plt.show()
# print(rel_diff)

# STEPS TO DO
# running convergence for the grid (tells us what nt for each)
# use convergence to see strong variations, make a mesh of points
# # then run the convergence with the right parameters

# # make the kappa/gamma and g/gamma ranges (n^2 points for now, make n and m different?)
# n_axis=9
# kappa_vals = np.logspace(-2,6,n_axis)
# g_vals = np.logspace(-2,6,n_axis)
# converged_nt = np.zeros((n_axis,n_axis))
# nt_test = np.array([200,400,800,1600,3200,6400])
# for i,kappa_temp in enumerate(kappa_vals):
#     for j, g_temp in enumerate(g_vals):
#         print(f"Doing point kappa={kappa_temp:.3e}, g={g_temp:.3e} ...")
#         beta_last = 0
#         for k in nt_test:
#             t,beta_t = beta_function(gamma=1,g=g_temp,kappa=kappa_temp,nt=k)
#             rel_diff = np.abs(beta_t[-1]-beta_last)/np.abs(beta_t[-1])
#             beta_last = beta_t[-1]
#             if rel_diff < 0.01:
#                 converged_nt[j,i] = k
#                 print(f"Converged at nt={k} (rel_diff={rel_diff:.3e})")
#                 break
# print(converged_nt)

# use itertools and parallelmap to compute using all cores. 