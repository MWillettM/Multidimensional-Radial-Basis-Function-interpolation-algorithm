<h1 align="center">Faul–Goodsell–Powell (FGP) Algorithm</h1>
<p align="center"><em>Fast NumPy implementation to generate an RBF interpolant in high dimensions</em></p>

---

## ✨ Features
- Core solver returns interpolation coefficients (`lambdas`) and constant (`alpha`).
- Supports multiple radial basis function (RBF) kernels: **multiquadric, Gaussian, inverse multiquadric, inverse quadratic**.
- Reproducible runs with **seed control** (`np.random.default_rng`).
- Helper functions to generate sample points from balls, cubes, grids, or Gaussian distributions.
- Helper function `interp` to evaluate the interpolant.
- Returns per-iteration diagnostics (`helpful_stats`) for analysis.
- The interpolant generated, s(x), is of the form 

    $s(x) = \sum_i^n \lambda_i \phi(\|x-x_i\|) + \alpha$

  where $\phi$ is your chosen kernel function.
---

## 📦 Requirements
- Python 3.9+
- `numpy`, `pandas`

---

## 🚀 Quick Start
```python
import numpy as np
from helper_functions import points_in_unit_ball, mq
from fgp_general import FGP

# Sample data
centers = points_in_unit_ball(100, 2, seed=42)
values = np.random.default_rng(42).uniform(0, 1, 100)

# Run FGP
num_iterations, lambdas, alpha, err, stats = FGP(
    data=centers, values=values,
    c=0.1, q=20, error=1e-5,
    max_iterations=1000, rbf_function=mq
)
print(num_iterations, err)
```

---

## 🧮 API
```python
FGP(data, values, c=0.1, q=30, error=1e-5, seed=42,
    max_iterations=1000, rbf_function=mq)
```

**Args**
- `data`: `(n, d)` array – interpolation centres  
- `values`: `(n,)` array – function values  
- `c`: float – RBF shape parameter  
- `q`: int – neighborhood size (20–50 typical)  
- `error`: float – stopping tolerance  
- `seed`: int – RNG seed  
- `max_iterations`: int – iteration cap  
- `rbf_function`: callable – one of `mq`, `gaussian`, `inv_mq`, `inv_quadratic`  

**Returns**
- `iterations`: int – number of iterations  
- `lambdas`: ndarray – interpolation coefficients  
- `alpha`: float – interpolation constant  
- `error`: float – final residual  
- `stats`: pandas.DataFrame – per-iteration diagnostics  

---

## ⚙️ Notes
- `q` too small → unstable; too large → slower. Stick to 20–50, independent of dataset size.  
- NaNs/divergence usually mean poor kernel/shape parameter choice.  

---

## 📖 Citation
If you use this repo, please cite:

> A.C. Faul, G. Goodsell, M.J.D. Powell.  
> *A Krylov subspace algorithm for multiquadric interpolation in many dimensions.*  
> IMA Journal of Numerical Analysis, 25 (2005), 1–24.  
> [doi:10.1093/imanum/drh021](https://doi.org/10.1093/imanum/drh021)

---

