import numpy as np
import os
from utils import cell_list, read_configuration
from gsdio import extract_positions
from numpy.fft import fftn, fftshift
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import gsd.hoomd

def compute_s_3d(x, box, N_grid):
    assert x.ndim == 2 and x.shape[1] == 3, f"Expected x=(N,3), got {x.shape}"
    assert np.shape(box) == (3,), f"Expected box=(3,), got {np.shape(box)}"
    N = x.shape[0]
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

def compute_partial_s_3d(x, types, box, N_grid):
    assert x.ndim == 2 and x.shape[1] == 3, f"Expected x=(N,3), got {x.shape}"
    assert np.shape(box) == (3,), f"Expected box=(3,), got {np.shape(box)}"
    N = x.shape[0]
    box = np.asarray(box)

    L_grid = np.min(box) / N_grid
    N_grid_vec = np.round(box / L_grid).astype(int)
    L_grid = box / N_grid_vec

    # Cell list for all particles
    cells, _ = cell_list(x, box, rmax=L_grid)
    lincell = np.ravel_multi_index((cells[:, 0], cells[:, 1], cells[:, 2]), dims=N_grid_vec, order='F')

    # Split by type
    mask1 = types == 0
    mask2 = types == 1
    lincell_1 = lincell[mask1]
    lincell_2 = lincell[mask2]

    bins = np.arange(np.prod(N_grid_vec) + 1)

    # Histogram for each type
    xgrid_1, _ = np.histogram(lincell_1, bins=bins)
    xgrid_2, _ = np.histogram(lincell_2, bins=bins)

    xgrid_1 = xgrid_1.reshape(N_grid_vec, order='F')
    xgrid_2 = xgrid_2.reshape(N_grid_vec, order='F')

    N1 = np.sum(mask1)
    N2 = np.sum(mask2)

    E1 = fftshift(fftn(xgrid_1))
    E2 = fftshift(fftn(xgrid_2))

    S_3_11 = np.abs(E1)**2 / N1
    S_3_22 = np.abs(E2)**2 / N2
    S_3_12 = (E1 * np.conj(E2)).real / np.sqrt(N1 * N2)

    return S_3_11, S_3_22, S_3_12, N_grid_vec


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

    num_1, _, _  = binned_statistic(q_1, S_3_flat, bins=q_binedge,
                                    statistic='sum')
    cnt_1, _, _  = binned_statistic(q_1, S_3_flat, bins=q_binedge,
                                    statistic='count')
    return q_bin_centers, num_1, cnt_1

def compute_partial_s_1d(x, types, box, N_grid):
    """
    Compute 1D radially averaged partial structure factors S_11(q), S_22(q), S_12(q)
    from particle positions and types.

    Parameters
    ----------
    x : np.ndarray
        (N, 3) particle positions.
    types : np.ndarray
        (N,) array of particle types (e.g., 0 and 1).
    box : array-like
        (3,) simulation box dimensions.
    N_grid : int
        Number of grid points in the smallest box dimension.

    Returns
    -------
    q_bin_centers : np.ndarray
        Centers of q bins (|q| magnitudes).
    S_11_1d, S_22_1d, S_12_1d : np.ndarray
        Radially averaged partial structure factors.
    """
    # Compute 3D partial structure factors
    S_3_11, S_3_22, S_3_12, _ = compute_partial_s_3d(x, types, box, N_grid)

    # Compute q-vectors
    q_3_x, q_3_y, q_3_z = compute_q3_grid(x, box, N_grid)

    # Compute |q| and flatten
    q_1 = np.sqrt(q_3_x**2 + q_3_y**2 + q_3_z**2).ravel()
    S_3_11_flat = S_3_11.ravel()
    S_3_22_flat = S_3_22.ravel()
    S_3_12_flat = S_3_12.ravel()

    # Bin edges based on average dq
    dq = np.mean(2 * np.pi / np.array(box))
    q_binedge = np.arange(np.min(q_1), np.max(q_1) + dq, dq)
    q_bin_centers = 0.5 * (q_binedge[:-1] + q_binedge[1:])

    # Bin and average
    S_11_1d, _, _ = binned_statistic(q_1, S_3_11_flat, bins=q_binedge, statistic='mean')
    S_22_1d, _, _ = binned_statistic(q_1, S_3_22_flat, bins=q_binedge, statistic='mean')
    S_12_1d, _, _ = binned_statistic(q_1, S_3_12_flat, bins=q_binedge, statistic='mean')

    S11_sum, _, _ = binned_statistic(q_1, S_3_11_flat, bins=q_binedge, statistic='sum')
    S22_sum, _, _ = binned_statistic(q_1, S_3_22_flat, bins=q_binedge, statistic='sum')
    S12_sum, _, _ = binned_statistic(q_1, S_3_12_flat, bins=q_binedge, statistic='sum')
    cnt,     _, _ = binned_statistic(q_1, S_3_11_flat, bins=q_binedge, statistic='count')

    return q_bin_centers, S11_sum, S11_sum, S12_sum, cnt

class StructureFactor:
    def __init__(self, gsd_path, N_grid, types = None, frames='all'):
        self.gsd_path = gsd_path
        self.N_grid = N_grid
        self.frames = frames
        self.types = types
        self._extract_data()

    def _extract_data(self):
        self.txt_path = os.path.splitext(self.gsd_path)[0] + '.txt'
        extract_positions(self.gsd_path, self.txt_path)
        self.x, self.box = read_configuration(self.txt_path, self.frames)


    def _iter_frames(self, frames=None):
        """Yield (pos, box) for each requested frame index."""
        for x_frame, box_frame in zip(self.x, self.box):
            yield x_frame, box_frame

    def compute_s_3d(self):
        if self.types is None:
            s_accum, n = None, 0
            for x, box in self._iter_frames(self.frames):
                s_i = compute_s_3d(x, box, self.N_grid)
                if s_accum is None:
                    s_accum = np.zeros_like(s_i)
                s_accum += s_i
                n += 1
            return s_accum / n
        else:
            s11_accum, s22_accum, s12_accum = None, None, None
            n = 0
            for x, box in self._iter_frames(self.frames):
                s11_i, s22_i, s12_i, _ = compute_partial_s_3d(x, self.types, box, self.N_grid)
                if s11_accum is None:
                    s11_accum = np.zeros_like(s11_i)
                    s22_accum = np.zeros_like(s22_i)
                    s12_accum = np.zeros_like(s12_i)
                s11_accum += s11_i
                s22_accum += s22_i
                s12_accum += s12_i
                n += 1
            return s11_accum / n, s22_accum / n, s12_accum / n

    def compute_s_1d(self):
        num_total, cnt_total = None, None
        if self.types is None:
            for x, box in self._iter_frames(self.frames):
                q, num_i, cnt_i = compute_s_1d(x, box, self.N_grid)
                if num_total is None:
                    num_total = np.zeros_like(num_i)
                    cnt_total = np.zeros_like(cnt_i)
                num_total += num_i
                cnt_total += cnt_i
            S1d = np.divide(num_total, cnt_total,
                    out=np.zeros_like(num_total),
                    where=cnt_total > 0)
            print(f'Total frames processed: {len(self.x)}')
            return q, S1d

        else:
            S11_sum_total = None
            S22_sum_total = None
            S12_sum_total = None
            cnt_total     = None
            q_bin         = None
            nframes       = 0

            for x, box in self._iter_frames(self.frames):
                q, S11_sum_i, S22_sum_i, S12_sum_i, cnt_i = compute_partial_s_1d(
                    x, self.types, box, self.N_grid
                )
                if S11_sum_total is None:
                    # initialize accumulators on first frame
                    S11_sum_total = np.zeros_like(S11_sum_i)
                    S22_sum_total = np.zeros_like(S22_sum_i)
                    S12_sum_total = np.zeros_like(S12_sum_i)
                    cnt_total     = np.zeros_like(cnt_i)
                    q_bin         = q  # keep the bin centers from the first frame

                S11_sum_total += S11_sum_i
                S22_sum_total += S22_sum_i
                S12_sum_total += S12_sum_i
                cnt_total     += cnt_i
                nframes       += 1

            # safe divide to produce final per-bin averages
            s11 = np.divide(S11_sum_total, cnt_total, out=np.zeros_like(S11_sum_total), where=cnt_total > 0)
            s22 = np.divide(S22_sum_total, cnt_total, out=np.zeros_like(S22_sum_total), where=cnt_total > 0)
            s12 = np.divide(S12_sum_total, cnt_total, out=np.zeros_like(S12_sum_total), where=cnt_total > 0)

            print(f'Total frames processed: {nframes}')
            self.q_bin, self.s_11, self.s_22, self.s_12 = q_bin, s11, s22, s12
            return self.q_bin, self.s_11, self.s_22, self.s_12

    def compute_q3_grid(self, frame = 0):
        x0  = self.x[frame]
        box0 = self.box[frame]
        self.q3x, self.q3y, self.q3z = compute_q3_grid(x0, box0, self.N_grid)
        return self.q3x, self.q3y, self.q3z