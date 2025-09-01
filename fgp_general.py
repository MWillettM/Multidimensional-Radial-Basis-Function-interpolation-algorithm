import numpy as np
import pandas as pd
from helper_functions import points_in_unit_ball, step3, step4, step5, step8, mq, gaussian, inv_mq, inv_quadratic
import time


def FGP(data: list,                       # List of data centers for interpolation
        values: list,                     # List of associated values for each center
        c: float = 0.1,                   # Shape parameter for your RBF
        q: int = 30,                      # Parameter for the FGP algorithm
        error: float = 1e-5,                     
        seed: int = 42,
        max_iterations: int = 1000,
        rbf_function = mq,                # See below for list of possible inputs.
        ):               

    """
    Runs the Faul–Goodsell–Powell (FGP) algorithm to build an RBF interpolant.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_dims)
        Interpolation centres.
    values : array-like of shape (n_samples,)
        Target values at the centres.
    c : float, default=0.1
        Shape parameter for the chosen RBF.
    q : int, default=30
        Local neighbourhood size used by FGP (typically 20–50).
    error : float, default=1e-5
        Stopping tolerance on the maximum absolute residual at the centres.
    seed : int, default=42
        Random seed for reproducibility.
    max_iterations : int, default=1000
        Hard cap on iterations.
    rbf_function : callable, default=mq
        rbfs included are: mq, gaussian, inv_mq and inv_quadratic

    Returns
    -------
    k : int
        Number of iterations performed.
    lambdas : ndarray of shape (n_samples,)
        Interpolation coefficients.
    alpha : float
        Interpolation constant.
    err : float
        Final maximum absolute residual at the centres.
    helpful_stats : pandas.DataFrame
        Per-iteration diagnostics (e.g. iteration, coefficients, alpha, error).
    """

    # -------------------- SETUP -------------------------

    # Reproducibility
    rng = np.random.default_rng(seed)

    # Setup depending on if the interpolation matrix is SPD or not. 
    if rbf_function in [gaussian, inv_mq, inv_quadratic]:
        pos_def_const = -1
    elif rbf_function == mq:
        pos_def_const = 1
    else:
        raise ValueError('Make sure your rbf_function is one of [gaussian, mq, inv_mq, inv_quadratic]')

    x_i = np.array(data)
    f_i = pos_def_const * np.array(values)
    n, d = x_i.shape

    # Checking that your data is the right size
    if n != len(f_i):
        raise ValueError('You must have the same number of centers and associated function values')

    start_time = time.time()

    # Set up intital values for the interpolation coefficients and interpolation constant, lambdas and alpha
    lambdas = np.zeros(n)
    alpha = 0.5 * (np.max(f_i) + np.min(f_i))

    # Calculating the initial residual.
    residual = f_i - np.ones(n) * alpha

    # Randomly shuffling the order of the datapoints to avoid combinations that result in slow convergence.
    omeg = rng.permutation(n) + 1
    data = []

    # Setting up the interpolation matrix.
    diffs = x_i.reshape((n, 1, d)) - x_i.reshape((1, n, d))
    squared_distance_matrix = (diffs**2).sum(axis=2)
    phi = pos_def_const * rbf_function(squared_distance_matrix, c)

    # step3 returns the 'lsets' - approximations for the q nearest neighbours for all but one point.
    for m in range(1, n):
        newomeg, lset, lvalue = step3(omeg, squared_distance_matrix, q, m, n)
        omeg = newomeg
        data.append([lvalue, lset, step4(lset,
                                         x_i,
                                         c,
                                         rbf_function,
                                         pos_def_const)])

    data = sorted(data, key=lambda x: x[0])

    # -------------------- SETUP COMPLETE -------------------------
    # -------------------- ITERATION BEGINS -------------------------

    # Iteration counter
    num_iterations = 0

    # First error value after setup
    err = np.max(np.abs(residual))

    # Required for iteration
    prev_direction = None

    # Tracking various helpful things across iterations
    helpful_stats = []

    # Iteration
    while err > error:
        num_iterations += 1
        tau = np.sum(np.array([step5(dat[1], dat[2], residual) for dat in data]), axis = 0)

        if num_iterations == 1:
            delta = tau.copy()

        else:
            beta = np.dot(tau, prev_direction) / (np.dot(delta, prev_direction) + 1e-10)
            delta = tau - beta * delta

        direction = phi @ delta
        prev_direction = direction.copy()
        lambdas, alpha, residual, err = step8(lambdas, alpha, residual, direction, delta)

        if np.isnan(err):
            print('Error is NaN, looks like the algoorithm has diverged')
            break

        # Tracking error values and alpha
        helpful_stats.append({'iteration': num_iterations, 'interp_coeffs': lambdas, 'interp_constant':alpha, 'error': err})

        if num_iterations > max_iterations:
            print(f'The algorithm did not converge within {max_iterations} iterations - tune your parameters.')
            break
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds in {num_iterations} iterations")

    helpful_stats = pd.DataFrame(helpful_stats)
    return num_iterations, lambdas, alpha, err, helpful_stats


# --------------- DEMO -------------------
if __name__ == '__main__':

    n = 100
    rng = np.random.default_rng(42)
    data = points_in_unit_ball(n, d = 2, seed = 42)
    values = rng.uniform(0, 1, n) 
    c = 0.1
    q = 20
    
    a,b,c,d, stats = FGP(data,
                    values,
                    0.1,
                    q,
                    error = 1e-5,
                    max_iterations = 1000,
                    rbf_function = inv_quadratic
                    )
    

