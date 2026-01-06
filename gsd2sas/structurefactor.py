import numpy as np
import os
import torch
from utils import read_configuration
from gsdio import extract_positions

def _torch_cell_list(x_t, box_t, rmax):
    rmax_t = torch.broadcast_to(rmax, box_t.shape)
    ncell = torch.floor(box_t / rmax_t).to(torch.int64)
    ncell = torch.clamp(ncell, min=3)

    x_wrapped = torch.remainder(x_t, box_t)
    cell = torch.floor(x_wrapped * ncell.to(x_t.dtype) / box_t).to(torch.int64)
    return cell, ncell

def _torch_binned_sum_count(values, bins, bin_edges):
    idx = torch.bucketize(bins, bin_edges) - 1
    nbins = bin_edges.shape[0] - 1
    valid = (idx >= 0) & (idx < nbins)
    idx = idx[valid]
    vals = values[valid]

    sums = torch.zeros((nbins,), dtype=values.dtype, device=values.device)
    counts = torch.zeros((nbins,), dtype=values.dtype, device=values.device)
    if vals.numel() > 0:
        sums.scatter_add_(0, idx, vals)
        counts.scatter_add_(0, idx, torch.ones_like(vals))
    return sums, counts

def _compute_s_3d_torch(x, box, N_grid, device=None, dtype=None):
    assert x.ndim == 2 and x.shape[1] == 3, f"Expected x=(N,3), got {x.shape}"
    assert np.shape(box) == (3,), f"Expected box=(3,), got {np.shape(box)}"
    if dtype is None:
        dtype = torch.float64
    x_t = torch.as_tensor(x, dtype=dtype, device=device)
    box_t = torch.as_tensor(box, dtype=dtype, device=device)
    N = x_t.shape[0]

    L_grid = torch.min(box_t) / N_grid
    n_grid_vec = torch.round(box_t / L_grid).to(torch.int64)
    L_grid = box_t / n_grid_vec.to(dtype)

    cells, _ = _torch_cell_list(x_t, box_t, rmax=L_grid)
    shape = (int(n_grid_vec[0]), int(n_grid_vec[1]), int(n_grid_vec[2]))
    xgrid_re = torch.zeros(shape, dtype=dtype, device=x_t.device)
    if N > 0:
        ones = torch.ones((N,), dtype=dtype, device=x_t.device)
        xgrid_re.index_put_((cells[:, 0], cells[:, 1], cells[:, 2]), ones, accumulate=True)

    S_3 = torch.abs(torch.fft.fftshift(torch.fft.fftn(xgrid_re))) ** 2 / float(N)

    return S_3, n_grid_vec

def compute_s_3d(x, box, N_grid, device=None, dtype=None):
    S_3, n_grid_vec = _compute_s_3d_torch(x, box, N_grid, device=device, dtype=dtype)
    return S_3.cpu().numpy(), n_grid_vec.cpu().numpy()

def _compute_partial_s_3d_torch(x, types, box, N_grid, device=None, dtype=None):
    assert x.ndim == 2 and x.shape[1] == 3, f"Expected x=(N,3), got {x.shape}"
    assert np.shape(box) == (3,), f"Expected box=(3,), got {np.shape(box)}"
    if dtype is None:
        dtype = torch.float64
    x_t = torch.as_tensor(x, dtype=dtype, device=device)
    box_t = torch.as_tensor(box, dtype=dtype, device=device)
    types_t = torch.as_tensor(types, device=x_t.device)

    L_grid = torch.min(box_t) / N_grid
    n_grid_vec = torch.round(box_t / L_grid).to(torch.int64)
    L_grid = box_t / n_grid_vec.to(dtype)

    cells, _ = _torch_cell_list(x_t, box_t, rmax=L_grid)
    shape = (int(n_grid_vec[0]), int(n_grid_vec[1]), int(n_grid_vec[2]))

    mask1 = types_t == 0
    mask2 = types_t == 1
    cells1 = cells[mask1]
    cells2 = cells[mask2]

    xgrid_1 = torch.zeros(shape, dtype=dtype, device=x_t.device)
    xgrid_2 = torch.zeros(shape, dtype=dtype, device=x_t.device)
    if cells1.numel() > 0:
        ones1 = torch.ones((cells1.shape[0],), dtype=dtype, device=x_t.device)
        xgrid_1.index_put_((cells1[:, 0], cells1[:, 1], cells1[:, 2]), ones1, accumulate=True)
    if cells2.numel() > 0:
        ones2 = torch.ones((cells2.shape[0],), dtype=dtype, device=x_t.device)
        xgrid_2.index_put_((cells2[:, 0], cells2[:, 1], cells2[:, 2]), ones2, accumulate=True)

    N1 = mask1.sum().to(dtype)
    N2 = mask2.sum().to(dtype)

    E1 = torch.fft.fftshift(torch.fft.fftn(xgrid_1))
    E2 = torch.fft.fftshift(torch.fft.fftn(xgrid_2))

    S_3_11 = torch.abs(E1) ** 2 / N1
    S_3_22 = torch.abs(E2) ** 2 / N2
    S_3_12 = torch.real(E1 * torch.conj(E2)) / torch.sqrt(N1 * N2)

    return S_3_11, S_3_22, S_3_12, n_grid_vec

def compute_partial_s_3d(x, types, box, N_grid, device=None, dtype=None):
    S_3_11, S_3_22, S_3_12, n_grid_vec = _compute_partial_s_3d_torch(
        x, types, box, N_grid, device=device, dtype=dtype
    )
    return (
        S_3_11.cpu().numpy(),
        S_3_22.cpu().numpy(),
        S_3_12.cpu().numpy(),
        n_grid_vec.cpu().numpy(),
    )


def _compute_q3_grid_torch(box, N_grid, device=None, dtype=None):
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

    if dtype is None:
        dtype = torch.float64
    box_t = torch.as_tensor(box, dtype=dtype, device=device)

    rmax = torch.min(box_t) / N_grid
    _, ncell = _torch_cell_list(torch.zeros((1, 3), dtype=dtype, device=box_t.device), box_t, rmax=rmax)

    dq = 2 * np.pi / box_t

    qx = torch.fft.fftshift(torch.fft.fftfreq(int(ncell[0]), d=1.0 / int(ncell[0]), device=box_t.device)) * dq[0]
    qy = torch.fft.fftshift(torch.fft.fftfreq(int(ncell[1]), d=1.0 / int(ncell[1]), device=box_t.device)) * dq[1]
    qz = torch.fft.fftshift(torch.fft.fftfreq(int(ncell[2]), d=1.0 / int(ncell[2]), device=box_t.device)) * dq[2]

    q3x, q3y, q3z = torch.meshgrid(qx, qy, qz, indexing='ij')
    return q3x, q3y, q3z

def compute_q3_grid(x, box, N_grid, device=None, dtype=None):
    q3x, q3y, q3z = _compute_q3_grid_torch(box, N_grid, device=device, dtype=dtype)
    return q3x.cpu().numpy(), q3y.cpu().numpy(), q3z.cpu().numpy()

def compute_s_1d(x, box, N_grid, device=None, dtype=None):

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
    if dtype is None:
        dtype = torch.float64
    S_3_t, _ = _compute_s_3d_torch(x, box, N_grid, device=device, dtype=dtype)
    q_3_x_t, q_3_y_t, q_3_z_t = _compute_q3_grid_torch(box, N_grid, device=device, dtype=dtype)
    q_1 = torch.sqrt(q_3_x_t ** 2 + q_3_y_t ** 2 + q_3_z_t ** 2).reshape(-1)

    S_3_t = S_3_t.reshape(-1)
    box_t = torch.as_tensor(box, dtype=dtype, device=device)
    dq = torch.mean(2 * np.pi / box_t)
    q_min = torch.min(q_1)
    q_max = torch.max(q_1)
    q_binedge = torch.arange(q_min, q_max + dq, dq, dtype=dtype, device=device)
    q_bin_centers = 0.5 * (q_binedge[:-1] + q_binedge[1:])

    num_1, cnt_1 = _torch_binned_sum_count(S_3_t, q_1, q_binedge)

    return q_bin_centers.cpu().numpy(), num_1.cpu().numpy(), cnt_1.cpu().numpy()

def compute_partial_s_1d(x, types, box, N_grid, device=None, dtype=None):
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
    if dtype is None:
        dtype = torch.float64
    S_3_11_t, S_3_22_t, S_3_12_t, _ = _compute_partial_s_3d_torch(
        x, types, box, N_grid, device=device, dtype=dtype
    )
    q_3_x_t, q_3_y_t, q_3_z_t = _compute_q3_grid_torch(box, N_grid, device=device, dtype=dtype)
    q_1 = torch.sqrt(q_3_x_t ** 2 + q_3_y_t ** 2 + q_3_z_t ** 2).reshape(-1)

    S_3_11_t = S_3_11_t.reshape(-1)
    S_3_22_t = S_3_22_t.reshape(-1)
    S_3_12_t = S_3_12_t.reshape(-1)

    box_t = torch.as_tensor(box, dtype=dtype, device=device)
    dq = torch.mean(2 * np.pi / box_t)
    q_min = torch.min(q_1)
    q_max = torch.max(q_1)
    q_binedge = torch.arange(q_min, q_max + dq, dq, dtype=dtype, device=device)
    q_bin_centers = 0.5 * (q_binedge[:-1] + q_binedge[1:])

    S11_sum, cnt = _torch_binned_sum_count(S_3_11_t, q_1, q_binedge)
    S22_sum, _ = _torch_binned_sum_count(S_3_22_t, q_1, q_binedge)
    S12_sum, _ = _torch_binned_sum_count(S_3_12_t, q_1, q_binedge)

    return (
        q_bin_centers.cpu().numpy(),
        S11_sum.cpu().numpy(),
        S22_sum.cpu().numpy(),
        S12_sum.cpu().numpy(),
        cnt.cpu().numpy(),
    )

class StructureFactor:
    def __init__(self, gsd_path, N_grid, types=None, frames='all', device=None, dtype=None):
        self.gsd_path = gsd_path
        self.N_grid = N_grid
        self.frames = frames
        self.types = types
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float64
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
                s_i, _ = compute_s_3d(x, box, self.N_grid, device=self.device, dtype=self.dtype)
                if s_accum is None:
                    s_accum = np.zeros_like(s_i)
                s_accum += s_i
                n += 1
            return s_accum / n
        else:
            s11_accum, s22_accum, s12_accum = None, None, None
            n = 0
            for x, box in self._iter_frames(self.frames):
                s11_i, s22_i, s12_i, _ = compute_partial_s_3d(
                    x, self.types, box, self.N_grid, device=self.device, dtype=self.dtype
                )
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
                q, num_i, cnt_i = compute_s_1d(x, box, self.N_grid, device=self.device, dtype=self.dtype)
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
                    x, self.types, box, self.N_grid, device=self.device, dtype=self.dtype
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
        self.q3x, self.q3y, self.q3z = compute_q3_grid(
            x0, box0, self.N_grid, device=self.device, dtype=self.dtype
        )
        return self.q3x, self.q3y, self.q3z
