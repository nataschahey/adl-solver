# Risk-Based Auto-Deleveraging Solver

Companion code for the paper:
> *Risk-Based Auto-Deleveraging*
>  Steven Campbell, Natascha Hey, Marcel Nutz, Ciamac Moallemi (2026)
> [arXiv link when available]

## Overview

This repository implements the numerical solver for the optimal ADL 
(auto-deleveraging) allocation problem in cross-margin cryptocurrency 
exchanges, as described in Section 4 of the paper. 

Given an aggregate reduction vector $Q$ and $n$ cross-margin accounts, 
the solver minimizes the expected exchange loss:

$$\min_{x \in \mathcal{X}} \mathbb{E}[\mathcal{L}(x, p_T)]$$

via dual gradient ascent with exact line search. Two price models are supported:

- **One-factor model**: population expectation computed in closed form
- **Bivariate GBM**: population expectation computed via 2D Gauss-Hermite quadrature

## Repository Structure
```
adl-solver/
├── adl/
│   ├── expectations.py   # analytical and quadrature expectation layer
│   ├── solver.py         # main solver: solve_adl, solve_water_filling
│   └── utils.py          # leverage helpers
├── examples/
│   └── numerical_example.ipynb   # reproduces Section 4 of the paper
├── requirements.txt
└── setup.py
```

## Installation
```bash
git clone https://github.com/your-username/adl-solver.git
cd adl-solver
pip install -r requirements.txt
pip install -e .
```

## Quick Start
```python
from adl import solve_adl
import numpy as np

# aggregate reduction vector
Q = np.array([10.0, 0.0])   # reduce 10 BTC, no ETH

# solve under one-factor model
x, obj, lam, success = solve_adl(Q, q, p_entry, p_tau, m,
                                  mode='one_factor', v=v)

# solve under full bivariate GBM
x, obj, lam, success = solve_adl(Q, q, p_entry, p_tau, m,
                                  mode='gbm',
                                  sigma=sigma_ann, Delta=Delta, rho=rho)
```

## Reproducing the Paper's Numerical Example
```bash
cd examples
jupyter notebook numerical_example.ipynb
```

## Requirements

- Python 3.8+
- numpy
- scipy
- matplotlib
- jupyter

## Citation

If you use this code, please cite:
```bibtex
<when available>
```

## License

MIT License
