import numpy as np

# ----------------  RBF FUNCTIONS -------------------

def mq(squared_distance_matrix, c):
    '''
    Returns the multiquadric function of the matrix of squared distances
    between pairs of points with shape parameter c
    '''
    return np.sqrt(c**2 + squared_distance_matrix)

def inv_mq(squared_distance_matrix, c):
    '''
    Returns the inverse multiquadric function of the matrix of squared distances
    between pairs of points with shape parameter c
    '''
    return 1 / np.sqrt(c**2 + squared_distance_matrix)

def gaussian(squared_distance_matrix, c):
    '''
    Returns the Gaussian function of the matrix of squared distances
    between pairs of points with shape parameter c
    '''
    return np.exp(-c * squared_distance_matrix)

def inv_quadratic(squared_distance_matrix, c):
    '''
    Returns the inverse quadratic function of the matrix of squared distances
    between pairs of points with shape parameter c
    '''
    return 1 / (c**2 + squared_distance_matrix)

# ---------------  FGP ALGORITHM STEP 3, 4, 5 and 8 -------------------

def step3(omeg,
          distances,
          q,
          m,
          n):
    omeg1 = np.array(omeg) 
    ell = omeg1[m-1]
    
    if n - m + 1 > q:
        while True:
            dist_2_ell = np.zeros(n - m + 1)
            for j in range(m, n + 1):
                jj = omeg1[j - 1]
                dist_2_ell[j - m] = distances[ell - 1, jj - 1]

            jj_indices = omeg1[m-1:n] - 1  
            dist_2_ell = distances[ell - 1, jj_indices] 
            
            # Get `q` nearest indices
            # Sorted_indices = np.argsort(dist_2_ell)[:q]  
            # Lset = omeg1[m-1:n][Sorted_indices]  
            idx_q = np.argpartition(dist_2_ell, q-1)[:q]
            Lset = omeg1[m-1:n][idx_q]

            # Compute pairwise distances for `Lset` (q × q symmetric matrix)
            Lset_idx = Lset - 1  
            dist_matrix = distances[Lset_idx[:, None], Lset_idx]  
            
            # Mask diagonal to ignore self-distances and find the minimum
            np.fill_diagonal(dist_matrix, np.inf) 
            min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            minbeta, mingamma = Lset[min_idx[0]], Lset[min_idx[1]]
            mindist = dist_matrix[min_idx]

            # Compute mindist2 (threshold for swapping)
            mindist2 = 0.5 * np.min(dist_2_ell[1:])

            # Swap if needed
            if mindist < mindist2:
                if distances[mingamma - 1, ell - 1] < distances[minbeta - 1, ell - 1]:
                    minbeta, mingamma = mingamma, minbeta  # Swap
                
                mhat = np.where(omeg1 == minbeta)[0][0]
                omeg1[[m-1, mhat]] = omeg1[[mhat, m-1]]
                ell = minbeta
            else:
                break  
    else:
        Lset = omeg1[m-1:n]  # No loop needed if n-m+1 ≤ q

    return omeg1.tolist(), Lset.tolist(), ell

def step4(lset,
          x_values,
          c,
          rbf_function,
          pos_def_const,
          eps = 1e-10,
          max_tries = 10):
    size = len(lset)
    x_ell = np.array(x_values)[np.array(lset)-1]

    q, d = x_ell.shape
    diffs = x_ell.reshape((q, 1, d)) - x_ell.reshape((1, q, d))
    squared_distance_matrix = (diffs**2).sum(axis=2)
    z_matrix = pos_def_const* rbf_function(squared_distance_matrix, c)

    z_matrix_padded = np.ones((size+1, size+1))
    z_matrix_padded[:size, :size] = z_matrix
    z_matrix_padded[size, size] = 0  

    dirac_vector = np.zeros(size + 1)
    dirac_vector[0] = 1.0

    # When d = 1, the padded z_matrix can become so poorly conditioned that linalg.solve will fail.
    # We cheat slightly by padding the nonzero diagonal entries if required.
    # We keep padding by larger and larger entries for 'max_tries' attempts
    for _ in range(max_tries):
        try:
            zeta = np.linalg.solve(z_matrix_padded, dirac_vector)
            zeta = zeta[:-1]
            zeta[-1] = -np.sum(zeta[:-1])
            return zeta
        except np.linalg.LinAlgError:
            z_matrix_padded[range(size), range(size)] += eps
            eps *= 10.0
    raise np.linalg.LinAlgError(
        f'step4 failed after {max_tries} attempts, the padded z_matrix is too poorly conditioned'
    )

def step5(lset, zeta, r):
    n = len(lset)
    taudummy = np.zeros(len(r))
    sum = 0
    sum = np.dot(zeta[:n], r[np.array(lset[:n])-1])
    myuell = sum / (zeta[0] + 1e-10)
    taudummy[np.array(lset[:n])-1] = myuell * zeta[:n]

    return taudummy


def step8(lambdas, alpha, r, d, delta):
    gamma = np.dot(delta, r) / (np.dot(delta, d) + 1e-10)
    r1 = r - gamma * d
    err = np.max(np.abs(r1))
    c = 0.5 * (np.max(r1) + np.min(r1))
    alpha1 = alpha + c
    r2 = r1 - c * np.ones(len(r))
    lambdas1 = lambdas + gamma * delta

    return lambdas1, alpha1, r2, err


# ------------ POINT GENERATORS ------------------

def points_in_unit_ball(n,
                        d,
                        seed = None
                        ):
    """
    Sample n points uniformly in the d-dimensional unit ball.

    If rng is None, one will be created from `seed` (or from entropy if seed=None).
    """
    rng = np.random.default_rng(seed)
    cube = rng.standard_normal(size=(n, d))
    norms = np.linalg.norm(cube, axis=1)
    surface_sphere = cube / norms[:, np.newaxis]
    scales = rng.uniform(0, 1, size=n)
    points = surface_sphere * (scales[:, np.newaxis]) ** (1 / d)
    return points


def points_in_cube(n_points, d):

    '''
    Generates n d-dimensional points uniformly distributed in the d-dimensional unit cube
    '''
    return np.random.uniform(0, 1, size=(n_points, d))


def points_normal(n_points, dimension):
    '''
    Returns n_points normally distributed points with dimension dimension.
    '''
    return np.random.normal(0, 1 / n_points, size=(n_points, dimension))


def points_integer_grid(n_points, d, s_min, s_max):
    '''
    Returns n points random sampled from the d-dimensional unit grid
    '''
    return np.random.randint(s_min, s_max, size=(n_points, d))

# ------------- INTERPOLANT EVALUATOR --------------

def interp(input_vec,       # Input vector - the point at which we want to evaluate the interpolant
            lambda_vec,     # Value of lambda (i.e. the vector of interpolation coefficients)
            alpha,          # Value of alpha  (i.e. the interpolation constant)
            centers_vec,    # Vector of the interpolation centers
            c,              # Shape parameter
            rbf_func = mq):            
    '''
    Evaluates the interpolant at a given point (input_vec). Returns the interpolated function value

    Make sure that the rbf_func is the same as the one used to generate the interpolation coefficients.
    '''    
    radial_shifts = np.array([rbf_func(np.abs(input_vec - i), c) for i in centers_vec])
    lambda_vec = np.array(lambda_vec)
    interp = np.dot(lambda_vec, radial_shifts) + alpha
    return interp
