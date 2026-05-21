import numpy as np
from numba import njit

#################### GEOMETRY (0-based Indexing) ####################
@njit
def pos_map(i, j, l):
    """Maps 2D coordinates (0 to l-1) to a 1D index."""
    return l * i + j

@njit
def posinvx(p, l):
    """Returns the x coordinate (0 to l-1) from a 1D index."""
    return p // l

@njit
def posinvy(p, l):
    """Returns the y coordinate (0 to l-1) from a 1D index."""
    return p % l

@njit
def neigh(p, l):
    """Finds the 4 nearest neighbors with periodic boundary conditions."""
    x = posinvx(p, l)
    y = posinvy(p, l)
    
    x_prev = (x - 1) % l
    x_next = (x + 1) % l
    y_prev = (y - 1) % l
    y_next = (y + 1) % l
    
    result = np.empty(4, dtype=np.int64)
    result[0] = pos_map(x_prev, y, l)
    result[1] = pos_map(x_next, y, l)
    result[2] = pos_map(x, y_prev, l)
    result[3] = pos_map(x, y_next, l)
    return result

@njit
def neigh_all(l):
    """Precomputes neighbors for all lattice sites."""
    n = l * l
    result = np.empty((n, 4), dtype=np.int64)
    for p in range(n):
        result[p] = neigh(p, l)
    return result

#################### INITIALISATION ####################
@njit
def init_spins(nspin, mode='random'):
    if mode == 'up':
        return np.ones(nspin, dtype=np.int64)
    elif mode == 'down':
        return -np.ones(nspin, dtype=np.int64)
    else:
        # Generates random +1 or -1
        return np.where(np.random.random(nspin) < 0.5, 1, -1)

@njit
def compute_Bloc(s, neighbours, nspin):
    """Sum of neighbor spins only — Bext excluded intentionally."""
    Bloc = np.empty(nspin, dtype=np.float64)
    for p in range(nspin):
        nb = neighbours[p]
        Bloc[p] = float(s[nb[0]] + s[nb[1]] + s[nb[2]] + s[nb[3]])
    return Bloc

@njit
def update_Bloc(Bloc, neighbours, ichange, old_spin):
    """O(4) incremental update: propagate spin change to neighbors."""
    nb = neighbours[ichange]
    delta = -2.0 * old_spin  
    for k in range(4):
        Bloc[nb[k]] += delta

#################### OBSERVABLES ####################
@njit
def compute_energy(s, Bloc, nspin, Bext):
    E_int = 0.0
    E_ext = 0.0
    for p in range(nspin):
        E_int += s[p] * Bloc[p]
        E_ext += s[p]
    return -0.5 * E_int - Bext * E_ext

@njit
def compute_magnetisation(s, nspin):
    M = 0.0
    for p in range(nspin):
        M += s[p]
    return M

#################### METROPOLIS ####################
@njit
def metropolis_step(s, Bloc, neighbours, nspin, beta, Bext):
    ichange = np.random.randint(0, nspin)
    s_i = s[ichange]
    
    dE = 2.0 * s_i * (Bloc[ichange] + Bext)
    if dE < 0 or np.random.random() < np.exp(-beta * dE):
        s[ichange] = -s_i
        update_Bloc(Bloc, neighbours, ichange, s_i)
        return dE, -2.0 * s_i
    return 0.0, 0.0

@njit
def metropolis_sweep(s, Bloc, neighbours, nspin, beta, Bext, n_steps):
    total_dE = 0.0
    total_dm = 0.0
    for _ in range(n_steps):
        dE, dm = metropolis_step(s, Bloc, neighbours, nspin, beta, Bext)
        total_dE += dE
        total_dm += dm
    return total_dE, total_dm

#################### MAIN SIMULATION ####################
@njit
def run_simulation(neighbours, nspin, n_mcs, beta, Bext, nthermal=0, snapshots=False):
    s = init_spins(nspin)
    Bloc = compute_Bloc(s, neighbours, nspin)
    E = compute_energy(s, Bloc, nspin, Bext)
    M = compute_magnetisation(s, nspin)

    energies = np.empty(n_mcs, dtype=np.float64)
    magnetisations = np.empty(n_mcs, dtype=np.float64)
    l = int(np.sqrt(nspin))

    if snapshots:
        s_history = np.empty((n_mcs, l, l), dtype=np.int64)
    else:
        s_history = np.empty((0, l, l), dtype=np.int64)

    # Thermalization phase
    for _ in range(nthermal):
        dE, dM = metropolis_sweep(s, Bloc, neighbours, nspin, beta, Bext, nspin)
        E += dE
        M += dM

    # Production phase
    for t in range(n_mcs):
        if snapshots:
            for i in range(l):
                for j in range(l):
                    s_history[t, i, j] = s[pos_map(i, j, l)]
                    
        dE, dM = metropolis_sweep(s, Bloc, neighbours, nspin, beta, Bext, nspin)
        E += dE
        M += dM
        energies[t] = E / nspin
        magnetisations[t] = M / nspin

    return energies, magnetisations, s_history

#################### ANALYSIS ####################
def observables(energies, magnetisations, beta, nspin):
    e = energies
    m = np.abs(magnetisations)

    e_mean = e.mean()
    m_mean = m.mean()
    e2_mean = (e**2).mean()
    m2_mean = (m**2).mean()
    m4_mean = (m**4).mean()

    C = beta**2 * nspin * (e2_mean - e_mean**2)
    chi = beta * nspin * (m2_mean - m_mean**2)
    binder = 1.0 - m4_mean / (3.0 * m2_mean**2) if m2_mean > 0 else 0.0

    return {
        "e_mean": float(e_mean),
        "m_mean": float(m_mean),
        "C": float(C),
        "chi": float(chi),
        "binder": float(binder),
    }