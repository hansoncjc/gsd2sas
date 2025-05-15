import numpy as np

def read_configuration(position_file, frame = 'all'):
    """
    Read the configuration from a text file created by io.extract_position.

    Parameters
    ----------
    position_file : str
        Path to the position file.
    frame : int or str, optional
        Total frame number to read starting from frame 0. If 'all', read all frames. Default is 'all'.
    Returns
    -------
    x : np.ndarray
        Particle positions. Shape: (F, N, 3), where F is the number of frames and N is the number of particles.
    box : np.ndarray
        Box dimensions. Shape: (3,)
    """
    
    # Set frame_max
    if frame == 'all':
        frame_max = np.inf
    else:
        frame_max = frame

    frame_count = 0

    # Read file
    x = []
    with open(position_file, 'r') as f:
        while frame_count < frame_max:
            line = f.readline()
            if not line:
                break

            N = int(line.strip())
            box = np.fromstring(f.readline().strip(), sep=' ')
            positions = []

            for _ in range(N):
                pos_line = f.readline()
                if not pos_line:
                    break
                positions.append([float(v) for v in pos_line.strip().split()])

            x.append(np.array(positions))
            frame_count += 1

            _ = f.readline()  # blank line

    return np.array(x), box
            
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