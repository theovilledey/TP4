import numpy as np
import matplotlib.pyplot as plt

# t = time grid
# t0 = pulse center
# sigma = pulse width (same for both photons)
# omega0 = frequency

def gaussian_wavepacket(t, t0, sigma, omega0):
    envelope = np.exp(-(t - t0) ** 2 / (2 * sigma ** 2))
    phase    = np.exp(1j * omega0 * (t - t0))
    norm     = (np.pi * sigma ** 2) ** (-0.25)
    return norm * envelope * phase

def overlap(tau, sigma, omega1, omega2, t_span=None):
    # normalised overlap integral of the two wavepackets
    if t_span is None:
        t_span = 30 * sigma
    t  = np.linspace(-t_span, t_span + abs(tau), 80_000)
    dt = t[1] - t[0]
    psi1 = gaussian_wavepacket(t, 0.0, sigma, omega1)
    psi2 = gaussian_wavepacket(t, tau, sigma, omega2)
    return np.sum(np.conj(psi1) * psi2) * dt

def hom_dip(tau_values, sigma, omega1, omega2):
    # curve 1/2*(1-overlap^2)
    coinc = []
    for tau in tau_values:
        g = overlap(tau, sigma, omega1, omega2)
        coinc.append(0.5 * (1.0 - abs(g) ** 2))
    return np.array(coinc)

def plot_wavepackets(ax,sigma=1.0,tau_example=1.5):
    # just plot the two wavepackets
    t = np.linspace(-6, 6 + tau_example, 2000)
    psi1 = gaussian_wavepacket(t,0,sigma,omega0=0)
    psi2 = gaussian_wavepacket(t,tau_example,sigma,omega0=0)
    ax.fill_between(t,np.abs(psi1)**2,alpha=0.35,color="tab:blue",label="Photon 1")
    ax.fill_between(t,np.abs(psi2)**2,alpha=0.35,color="tab:orange",label="Photon 2")
    ax.plot(t,np.abs(psi1)**2,color="tab:blue",lw=1.5)
    ax.plot(t,np.abs(psi2)**2,color="tab:orange",lw=1.5)
    ax.axvline(0,ls="--", color="tab:blue",lw=1, alpha=0.7)
    ax.axvline(tau_example, ls="--", color="tab:orange",lw=1, alpha=0.7)
    ax.set_xlabel(r"$t/\sigma$")
    ax.set_ylabel(r"$|\psi|^2$")
    ax.legend()
    ax.set_xlim(t[0], t[-1])

def fig_basic_dip():
    # plot the prob curve next to the wavepackets
    sigma = 1.0
    tau   = np.linspace(-6, 6, 300)
    coinc = hom_dip(tau, sigma, 0, 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_wavepackets(axes[0],sigma=sigma,tau_example=2.0)
    axes[1].plot(tau, coinc, color="tab:blue", lw=2.5)
    axes[1].axhline(0.5, color="gray", ls=":", lw=1, label="Classical limit")
    axes[1].fill_between(tau,coinc,0.5,where=(coinc<0.5),alpha=0.15,color="tab:blue")
    axes[1].set_xlabel(r"$\tau/\sigma$")
    axes[1].set_ylabel("Coincidence probability")
    axes[1].set_ylim(-0.02, 0.55)
    axes[1].legend()
    plt.tight_layout()
    plt.show()

def fig_frequency_mismatch():
    # coincidence as function of freq. mismatch
    sigma  = 1.0
    tau    = np.linspace(-6, 6, 300)
    deltas = [0.0, 1.0] # in rad/sigma
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:blue", "tab:orange"]
    for i, dw in enumerate(deltas):
        coinc = hom_dip(tau, sigma, 0, dw)
        ax.plot(tau, coinc, lw=2.2, color=colors[i],
                label=fr"$\Delta\omega = {dw}\ \mathrm{{rad}}/\sigma$")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Classical limit")
    ax.set_xlabel(r"$\tau/\sigma$")
    ax.set_ylabel("Coincidence probability")
    ax.set_ylim(-0.02, 0.55)
    ax.legend()
    plt.tight_layout()
    plt.show()

fig_basic_dip()
fig_frequency_mismatch()