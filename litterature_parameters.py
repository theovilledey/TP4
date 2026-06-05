import numpy as np

# for clarity I compute in ordinary frequency, and express as 2pi*f
# this program selects the best emitter fit on it's own
c = 2.99792458e8
Q = 44000 # quality factor
lambda_0 = 1513e-9 # cavity wavelength [m]
V_c = 6.5e-20 # cavity mode volume [m^3]
n = 2.5 # refractive index at cavity mode location

emitters = {
    'A': {'lambda': 650e-9, 'T_1': 0.7e-9, 'FWHM': 4e-9, 'gamma': 227e6},
    'B': {'lambda': 1350e-9, 'T_1': 0.2e-9, 'FWHM': 10e-9, 'gamma': 796e6} }

# pick closest emitter by wavelength
key = min(emitters, key=lambda k: abs(emitters[k]['lambda'] - lambda_0))
em = emitters[key]
T_1 = em['T_1']
f_gamma = em['gamma']
print(f"Selected emitter {key}")

f_0 = c/lambda_0
f_kappa = f_0/Q
f_g = np.sqrt(3*lambda_0**2*c/(8*np.pi*n**3*V_c*T_1))/(2*np.pi)

g_over_gamma = f_g/f_gamma
kappa_over_gamma = f_kappa/f_gamma

print(f"g/gamma = {g_over_gamma:.4f}")
print(f"kappa/gamma = {kappa_over_gamma:.4f}")