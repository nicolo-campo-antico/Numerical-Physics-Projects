import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from ising_core import neigh_all, run_simulation, observables

def main():
    l = 500
    nspin = l * l
    n_mcs = 200
    beta = 0.44 
    Bext = 0.0
    nthermal = 100
    
    print("Running simulation...")
    neighbours = neigh_all(l)
    energies, magnetisations, snaps = run_simulation(
         neighbours, nspin, n_mcs, beta, Bext, nthermal, snapshots=True
    )
    obs = observables(energies, magnetisations, beta, nspin)
    
    # CLI Table Output
    rows = [
         ("Mean energy/spin",          "⟨e⟩",   obs["e_mean"],  "J"  ),
         ("Mean magnetisation/spin",   "⟨|m|⟩", obs["m_mean"],  ""   ),
         ("Specific heat",             "C",      obs["C"],       "k_B"),
         ("Magnetic susceptibility",   "χ",      obs["chi"],     "1/J"),
         ("Binder cumulant",           "U₄",     obs["binder"],  ""   ),
    ]
    col_w = [30, 8, 12, 5]
    sep = "─" * (sum(col_w) + 3 * 3 + 2)
    header = f"{'Observable':<{col_w[0]}}   {'Sym':^{col_w[1]}}   {'Value':>{col_w[2]}}   {'Unit':<{col_w[3]}}"
    
    print(sep)
    print(header)
    print(sep)
    for desc, sym, val, unit in rows:
         print(f"{desc:<{col_w[0]}}   {sym:^{col_w[1]}}   {val:>{col_w[2]}.6f}   {unit:<{col_w[3]}}")
    print(sep)
    
    # Plotting and Animation
    t = np.arange(n_mcs)
    fig = plt.figure(figsize=(10, 5), facecolor="#0d1117")
    
    
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2], hspace=0.45, wspace=0.35)
    style = dict(facecolor="#0d1117")
    
    
    
    ax_g = fig.add_subplot(gs[:, 0], facecolor="#0d1117")
    im = ax_g.imshow(snaps[0], cmap="binary", vmin=-1, vmax=1, interpolation="nearest")
    ax_g.axis("off")
    
    
    ttl = ax_g.text(0.5, 1.02, "t = 0", color="white", fontsize=10, 
                     ha="center", transform=ax_g.transAxes)
    
    # energy Plot
    ax_e = fig.add_subplot(gs[0, 1], **style)
    ax_e.plot(t, energies, color="#00c8ff", lw=1.2)
    vline_e = ax_e.axvline(0, color="white", lw=0.8, ls="--", alpha=0.6)
    ax_e.set_ylabel("e / spin", color="white", fontsize=9)
    ax_e.set_title("Energy", color="white", fontsize=10)
    ax_e.tick_params(colors="gray", labelsize=8)
    for sp in ax_e.spines.values(): sp.set_color("#2a3a4a")
    
    # Magnetisation Plot
    ax_m = fig.add_subplot(gs[1, 1], **style)
    ax_m.plot(t, magnetisations, color="#ff4f7b", lw=1.2)
    vline_m = ax_m.axvline(0, color="white", lw=0.8, ls="--", alpha=0.6)
    ax_m.set_ylabel("m / spin", color="white", fontsize=9)
    ax_m.set_xlabel("MCS", color="white", fontsize=9)
    ax_m.set_title("Magnetisation", color="white", fontsize=10)
    ax_m.tick_params(colors="gray", labelsize=8)
    for sp in ax_m.spines.values(): sp.set_color("#2a3a4a")
    
    fig.suptitle(
         f"2D Ising  ·  L={l}  ·  β={beta}  ·  B={Bext}",
         color="white", fontsize=12, y=0.98
    )
    
    def update(frame):
         im.set_data(snaps[frame])
         ttl.set_text(f"t = {frame}")  
         vline_e.set_xdata([frame])
         vline_m.set_xdata([frame])
         return im, ttl, vline_e, vline_m
    
    
    ani = FuncAnimation(fig, update, frames=n_mcs, interval=80, blit=False, repeat=True)
    
    print("Saving GIF...")
    ani.save('ising_2d.gif', writer='pillow', fps=25, dpi=75)
    print("GIF Saved!")
    
    plt.show()
    
    

if __name__ == '__main__':
    main()