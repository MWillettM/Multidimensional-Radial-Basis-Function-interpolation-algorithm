**FAUL-GOODSEN-POWELL algorithm**

A robust NumPy implementation of the FGP algorithm for scattered, high-dimensional data.

The interpolant generated, s(x), is of the form 

$s(x) = \sum_i^n \lambda_i \phi(\|x-x_i\|) + \alpha$

✨ Features

The core function returns the interpolation coefficients and constant.

Supports multiple radial basis function (RBF) kernels: multiquadric, Gaussian, inverse multiquadric, inverse quadratic.

You can set a seed value for reproducibility.

Includes helper functions to generate example data sample from the d-dimensional (ball, cube, grid, Gaussian).

Includes helper function to evaluate your interpolant (interp).

Returns stats from throughout the run for diagnostics or analysis (helpful_stats)

📦 Requirements

Python 3.9+

numpy, pandas

🚀 Quick Start
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

🧮 API
FGP(data, values, c=0.1, q=30, error=1e-5, seed=42,
    max_iterations=1000, rbf_function=mq)

Args:

data: (n, d) array – interpolation centres

values: (n,) array – function values

c: RBF shape parameter

q: neighborhood size (20–50 typical)

error: stopping tolerance

seed: RNG seed

max_iterations: iteration cap

rbf_function: one of mq, gaussian, inv_mq, inv_quadratic

Returns:
(iterations, lambdas, alpha, error, stats)

🧪 Evaluate the Interpolant
from helper_functions import interp, mq
y_star = interp([0.1, 0.2], lambdas, alpha, centers, 0.1, mq)

⚙️ Notes

q too small → unstable; too large → slower. Stick tpo between 20 and 50, independent of your dataset size.

NaNs/divergence usually mean poor kernel/shape parameter choice.

📖 Citation

If you use this repo, please cite:

A.C. Faul, G. Goodsell, M.J.D. Powell.
A Krylov subspace algorithm for multiquadric interpolation in many dimensions.
IMA J. Numer. Anal. 25 (2005), 1–24. doi:10.1093/imanum/drh021

