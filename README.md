# Structure-Factor
Read .gsd files and plot extracted structure factor.

## Quick start
Example notebook:
- `examples/Sphere.ipynb`
- `examples/Sphere_pytorch.ipynb`

Unit tests:
```bash
python tests/single_sim_test.py
```

GPU benchmark (Hyak):
```bash
python tests/cpu_gpu_bench.py
```

## Frames selection
`StructureFactor(..., frames=...)` accepts:
- `'all'` (default)
- `int` for a single frame index
- iterable of `int` for specific frame indices
- `'last:N'` for the last N frames

## GPU usage
Use the torch-backed structure factor with explicit device/dtype:
```python
from gsd2sas.structurefactor import StructureFactor
import torch

sf = StructureFactor(gsd_path, N_grid, frames="last:10", device=torch.device("cuda"), dtype=torch.float64)
q, s = sf.compute_s_1d()
```
