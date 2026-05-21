# 2D Ising Model — Metropolis Monte Carlo

A Python reimplementation of the 2D Ising model simulations from the **Statistical Physics of Complex Systems** course (Prof. Biscari, Politecnico di Milano). The original algorithms — written in Mathematica — have been ported, optimized for performance, and modularized for clarity and reuse.

---

## Physics background

The 2D Ising model is a canonical system in statistical mechanics: a square lattice of spins $s_i = \pm 1$ interacting via the Hamiltonian

$$\mathcal{H} = -J\sum_{\langle i,j \rangle} s_i s_j - B\sum_i s_i$$

It exhibits a second-order phase transition at the exact critical temperature $\beta_c = \ln(1+\sqrt{2})/2 \approx 0.4407$ (Onsager, 1944). Near criticality, fluctuations diverge and observables like the specific heat $C$ and magnetic susceptibility $\chi$ peak sharply — all of which this code measures.

---

## Implementation highlights

| Feature | Detail |
|---|---|
| Algorithm | Metropolis–Hastings single-spin flip |
| Boundary conditions | Periodic (toroidal lattice) |
| Performance | JIT-compiled hot loops via **Numba** `@njit` |
| Observables | $\langle e \rangle$, $\langle\|m\|\rangle$, $C$, $\chi$, Binder cumulant $U_4$ |

---

## Project structure

```
├── ising_core.py         # Core library: geometry, Metropolis, observables
└── ising_2D_analysis.py  # Simulation driver, plots, GIF export
```

`ising_core.py` is designed to be imported independently — all performance-critical functions are Numba-compiled and have no matplotlib dependency.

---

## Usage

**Install dependencies:**
```bash
pip install numpy matplotlib numba pillow
```

**Run:**
```bash
python ising_2D_analysis.py
```

This runs a simulation with the default parameters (L=100, β=0.44, B=0, 200 MCS after 100 thermalization steps), prints observables to the terminal, and saves `ising_2d.gif`.

**Key parameters** in `ising_2D_analysis.py`:

```python
l        = 100    # Lattice side length (N = l²)
n_mcs    = 200    # Monte Carlo sweeps (production)
beta     = 0.44   # Inverse temperature (β_c ≈ 0.4407)
Bext     = 0.0    # External magnetic field
nthermal = 100    # Thermalization sweeps (discarded)
```

**Example terminal output:**
```
──────────────────────────────────────────────────────────
Observable                      Sym          Value   Unit
──────────────────────────────────────────────────────────
Mean energy/spin               ⟨e⟩       -1.320541   J
Mean magnetisation/spin       ⟨|m|⟩       0.038412
Specific heat                   C         42.18763   k_B
Magnetic susceptibility         χ        108.94201   1/J
Binder cumulant                U₄          0.11832
──────────────────────────────────────────────────────────
```

---

## Notes

- First run triggers Numba JIT compilation (~10–20 s). Subsequent runs are fast.
- The Binder cumulant $U_4 \to 2/3$ in the ordered phase and $\to 0$ in the disordered phase; its crossing point across lattice sizes locates $\beta_c$ precisely.
- To probe the phase transition, sweep `beta` in the range `[0.35, 0.55]` and track the peak of $\chi$ and $C$.
