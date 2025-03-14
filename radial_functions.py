import numpy as np


def MQ(x, c):
    k = np.sqrt(c**2 + np.linalg.norm(x) ** 2)
    return k


def Gaussian(point, c):
    return np.exp(-c * np.linalg.norm(point))


def points_normal(n, d):
    points = np.random.normal(0, 1 / n, size=(n, d))
    return points


# x is the input, l is the vector of values of lambda, a is the value of alpha, X is the vector of x_i, c is the shape parameter
def interp(x, l, a, X, c):
    radial_shifts = np.array([MQ(np.abs(x - i), c) for i in X])
    l = np.array(l)
    interp = np.dot(l, radial_shifts) + a
    return interp


def smallest_distance(x):
    min_distance = float("inf")
    for i in range(len(x) - 1):
        dist = x[i + 1][1] - x[i][1]
        if dist < min_distance:
            min_distance = dist
            Beta = i
            Gamma = i + 1
    return min_distance, Beta, Gamma


# X is the input vector, x is the value with which to evaluate d(x), d is delta, c is shape parameter - this returns d^k(x)
def search_direction(X, x, d, c):
    values = [MQ(x - i, c) for i in X]
    k = np.dot(values, d)
    return k


def step3(omeg, distances, q, m, n):
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
            Sorted_indices = np.argsort(dist_2_ell)[:q]  
            Lset = omeg1[m-1:n][Sorted_indices]  

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


def step4(lset, x_values, c):
    size = len(lset)
    x_ell = np.array(x_values)[np.array(lset)-1]

    n, d = x_ell.shape
    diffs = x_ell.reshape((n, 1, d)) - x_ell.reshape((1, n, d))
    squared_distance_matrix = (diffs**2).sum(axis=2)
    z_matrix = np.sqrt(squared_distance_matrix + c**2)

    z_matrix_padded = np.ones((size+1, size+1))
    z_matrix_padded[:size, :size] = z_matrix
    z_matrix_padded[size, size] = 0  

    dirac_vector = np.zeros(size + 1)
    dirac_vector[0] = 1

    zeta = np.linalg.solve(z_matrix_padded, dirac_vector)
    zeta = zeta[:-1]
    zeta[-1] = -np.sum(zeta[:-1])
    return zeta



def step5(lset, zeta, r):
    n = len(lset)
    taudummy = np.zeros(len(r))
    sum = 0
    
    sum = np.dot(zeta[:n], r[np.array(lset[:n])-1])

    myuell = sum / zeta[0]

    taudummy[np.array(lset[:n])-1] = myuell * zeta[:n]

    return taudummy


def step8(lambdas, alpha, r, d, delta):
    gamma = np.dot(delta, r) / np.dot(delta, d)
    r1 = r - gamma * d
    err = np.max(np.abs(r1))
    c = 0.5 * (np.max(r1) + np.min(r1))
    alpha1 = alpha + c
    r2 = r1 - c * np.ones(len(r))
    lambdas1 = lambdas + gamma * delta

    return lambdas1, alpha1, r2, err


# Different underlying distributions to draw from in the DEMO version.


def points_in_unit_ball(n, d):
    cube = np.random.standard_normal(size=(n, d))
    norms = np.linalg.norm(cube, axis=1)
    surface_sphere = cube / norms[:, np.newaxis]
    scales = np.random.uniform(0, 1, size=n)
    points = surface_sphere * (scales[:, np.newaxis]) ** (1 / d)
    return points


def points_in_cube(n, d):
    points = np.random.uniform(0, 1, size=(n, d))
    return points


def points_normal(n, d):
    points = np.random.normal(0, 1, size=(n, d))
    return points


def points_integer_grid(n, d, s_min, s_max):
    points = np.random.randint(s_min, s_max, size=(n, d))
    return points
