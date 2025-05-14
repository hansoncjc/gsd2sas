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
        Particle positions. Shape: (N, D)
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
            frame += 1

            _ = f.readline()  # blank line

    return np.array(x), box
            
