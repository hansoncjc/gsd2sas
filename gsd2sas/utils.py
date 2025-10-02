import numpy as np

def read_configuration(position_file, frames = 'all'):
    """
    Parameters
    ----------
    frames : int | Iterable[int] | 'all'
        Which frame indices to read from the file.
    Returns
    -------
    x   : (F, N, 3) float64   positions for each requested frame
    box : (F, 3)     float64   box lengths for each frame
    """
    
    if frames == 'all':
        read_all = True
        target = None
    elif isinstance(frames, int):
        read_all = False
        target = {frames}
    else:                           # iterable of ints
        read_all = False
        target = set(frames)


    x_list, box_list = [], []
    with open(position_file) as f:
        frame_idx = 0
        while True:
            first = f.readline()
            if not first: break
            N     = int(first)
            box   = np.fromstring(f.readline(), sep=' ')
            pos   = np.fromfile(f, count=3*N, sep=' ').reshape(N,3)
            #_     = f.readline()            # blank line
            #_     = f.readline()            # blank line

            if read_all or frame_idx in target:
                x_list.append(pos.copy())
                box_list.append(box.copy())
            frame_idx += 1
            if not read_all and frame_idx > max(target):
                break
    return np.stack(x_list), np.stack(box_list)
            
def cell_list(x, box, rmax):
    """
    Assign particles to 3D spatial cells.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (N, 3) with particle positions.
    box : array-like
        Box dimensions. Shape: (3,)
    rmax : float or array-like
        Maximum cell size per dimension (scalar or (3,) array).

    Returns
    -------
    cells : np.ndarray
        index of the cell to which each particle belongs
    Ncell : np.ndarray
        (3,) number of cells in each dimension.
    """
    box = np.asarray(box)
    rmax = np.broadcast_to(rmax, box.shape)

    Ncell = np.floor(box / rmax).astype(int)
    Ncell[Ncell < 3] = 3  # Ensure at least 3 cells per dimension

    # Apply periodic boundary conditions
    x_wrapped = np.mod(x, box)

    # Scale position to cell index (0-based indexing)
    cell = np.floor(x_wrapped * Ncell / box).astype(int)
    cells = np.squeeze(cell)

    return cells, Ncell

def load_types(types_file):
    return np.loadtxt(types_file)

