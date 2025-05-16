import numpy as np
from utils import cell_list
from numpy.fft import fftn, fftshift
from scipy.stats import binned_statistic

def compute_s_3d(x, box, N_grid):
    N = x.shape[1]
    box = np.asarray(box)

    # Determine grid resolution
    L_grid = np.min(box) / N_grid
    N_grid_vec = np.round(box / L_grid).astype(int)
    L_grid = box / N_grid_vec
    cells, _ = cell_list(x, box, rmax=L_grid)
    # Convert to linear indices
    lincell = np.ravel_multi_index((cells[:, 0], cells[:, 1], cells[:, 2]), dims=N_grid_vec, order='F')

    # Count number of particles in each cell
    bins = np.arange(np.prod(N_grid_vec) + 1)
    xgrid, _ = np.histogram(lincell, bins=bins)
    xgrid_re = xgrid.reshape(N_grid_vec, order='F')

    S_3 = np.abs(fftshift(fftn(xgrid_re)))**2 / N

    return S_3, N_grid_vec

def compute_q3_grid(x, box, N_grid):
    """
    Compute the 3D grid of q-vectors for a given configuration.
    
    Parameters
    ----------
    x : np.ndarray
        Particle positions. Shape: (N, 3).
    box : np.ndarray
        Box dimensions. Shape: (3,).
    N_grid : int
        number of grid points in the smallest dimension
    
    Returns
    -------
    q3x, q3y, q3z : float
        q-vectors in x, y, and z directions.
    """

    box = np.asarray(box)

    # Use cell_list to get Ncell (equivalent to N_grid_vec)
    _, Ncell = cell_list(x, box, rmax=np.min(box) / N_grid)

    # Compute reciprocal space step
    dq = 2 * np.pi / box

    # Reciprocal grid coordinates in each direction
    qx = np.fft.fftshift(np.fft.fftfreq(Ncell[0], d=1./Ncell[0])) * dq[0]
    qy = np.fft.fftshift(np.fft.fftfreq(Ncell[1], d=1./Ncell[1])) * dq[1]
    qz = np.fft.fftshift(np.fft.fftfreq(Ncell[2], d=1./Ncell[2])) * dq[2]

    q3x, q3y, q3z = np.meshgrid(qx, qy, qz, indexing='ij')
    return q3x, q3y, q3z

def compute_s_1d(x, box, N_grid):
    """
    Compute 1D radially averaged structure factor S(q) from particle positions.

    Parameters
    ----------
    x : np.ndarray
        (N, 3) particle positions.
    box : array-like
        (3,) simulation box lengths.
    N_grid : int
        Number of grid points in the smallest box dimension.

    Returns
    -------
    S_1 : np.ndarray
        1D radially averaged structure factor.
    q_1_centers : np.ndarray
        Centers of q bins (magnitude of wavevector).
    """
    # Compute 3D structure factor
    S_3, _= compute_s_3d(x, box, N_grid)

    # Compute q-vectors
    q_3_x, q_3_y, q_3_z = compute_q3_grid(x, box, N_grid)

    # Compute |q| and flatten
    q_1 = np.sqrt(q_3_x**2 + q_3_y**2 + q_3_z**2).ravel()
    S_3_flat = S_3.ravel()

    # Set bin width based on average q-resolution
    dq = np.mean(2 * np.pi / np.array(box))
    print(dq)
    q_bin_centers = np.arange(0, np.max(q_1) + dq, dq)
    q_binedge = np.concatenate([q_bin_centers, [q_bin_centers[-1] + dq]]) - dq / 2

    # Bin and average S values over spherical shells
    S_1, _, _ = binned_statistic(q_1, S_3_flat, bins=q_binedge, statistic='mean')

    return q_bin_centers, S_1