# Dynamical Systems

Simple Numerical simulation and visualization toolkit for dynamical systems using Python and SciPy.

This project solves ordinary differential equations (ODEs), analyzing system dynamics, and visualizing trajectories, phase spaces, and energy evolution for several classical physical systems.

---

## Features

- Numerical integration with `scipy.integrate.solve_ivp` (`RK45`, `DOP853`, `Radau`, ...)
- Modular architecture for adding custom dynamical systems
- Automatic energy computation for supported models
- Time-domain visualization
- Phase-space analysis
- Optional Poincaré sections

---

## Implemented Models

Currently included systems:

- Simple Pendulum
- Double Pendulum
- Duffing Oscillator

One may add its own Dynamical System with an arbitrary number of degree of freedom

---

## Example Outputs

The framework can generate:

- State evolution plots + Energy conservation analysis
- Phase-space trajectories
- Poincaré sections

