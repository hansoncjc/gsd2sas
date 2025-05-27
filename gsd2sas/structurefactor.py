import numpy as np
import os
from utils import cell_list, read_configuration
from gsdio import extract_positions
from numpy.fft import fftn, fftshift
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import gsd.hoomd

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
        Centers of qr bins (magnitude of wavevector, non-dimensional).
    """
    # Compute 3D structure factor
    S_3, _= compute_s_3d(x, box, N_grid)

    # Compute q-vectors
    q_3_x, q_3_y, q_3_z = compute_q3_grid(x, box, N_grid)

    # Compute |q| and flatten
    q_1 = np.sqrt(q_3_x**2 + q_3_y**2 + q_3_z**2).ravel()
    S_3_flat = S_3.ravel()
    # print(np.min(q_1), np.max(q_1))
    # Set bin width based on average q-resolution
    dq = np.mean(2 * np.pi / np.array(box))
    q_binedge = np.arange(np.min(q_1), np.max(q_1) + dq, dq)
    q_bin_centers = 0.5 * (q_binedge[:-1] + q_binedge[1:])


    # Bin and average S values over spherical shells
    S_1, _, _ = binned_statistic(q_1, S_3_flat, bins=q_binedge, statistic='mean')

    # Compute bin counts
    bin_counts, _, _ = binned_statistic(q_1, S_3_flat, bins=q_binedge, statistic='count')

    return q_bin_centers, S_1

class StructureFactor:
    def __init__(self, gsd_path, N_grid, frame='all'):
        self.gsd_path = gsd_path
        self.N_grid = N_grid
        self.frame = frame
        self._extract_data()

    def _extract_data(self):
        self.txt_path = os.path.splitext(self.gsd_path)[0] + '.txt'
        extract_positions(self.gsd_path, self.txt_path)
        self.x, self.box = read_configuration(self.txt_path, self.frame)

    def compute_s_3d(self):
        self.s_3d, _ = compute_s_3d(self.x, self.box, self.N_grid)
        return self.s_3d

    def compute_s_1d(self):
        self.q_bin, self.s_1d = compute_s_1d(self.x, self.box, self.N_grid)
        return self.q_bin, self.s_1d

    def compute_q3_grid(self):
        self.q3x, self.q3y, self.q3z = compute_q3_grid(self.x, self.box, self.N_grid)
        return self.q3x, self.q3y, self.q3z