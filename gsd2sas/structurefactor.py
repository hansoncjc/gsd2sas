import numpy as np
from utils import cell_list

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