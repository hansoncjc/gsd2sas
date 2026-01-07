import multiprocessing as mp
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.append('/mmfs1/gscratch/zeelab/hanson/codes/DNA_lipid_silica/gsd2sas/gsd2sas')
from structurefactor import StructureFactor
from sasintensity import SphereIntensity


GSD_PATH = "./N8000.0_phi0.010_epsd7.10_delta0.25_epsY0.27_lamb4.28.gsd"
N_GRID = 300
FRAMES = 'all'


def _run_device(label, device_str):
    device = torch.device(device_str)
    dtype = torch.float64

    print(f"[{label}] start on device={device}")

    au = SphereIntensity(volume_fraction=0.01, sld_sample=118e-6, sld_solvent=9.44e-6)
    au.set_form_factor(radius=1)

    au.set_structure_factor(gsd_path=GSD_PATH, N_grid=N_GRID, frames=FRAMES)
    au.structure_factor.device = device
    au.structure_factor.dtype = dtype

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    q, s = au.structure_factor.compute_s_1d()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    gsd_dir = Path(GSD_PATH).resolve().parent
    out_path = gsd_dir / f"structure_factor_{label}.png"

    plt.figure(dpi=120)
    plt.loglog(q[3:-1] / 100, s[3:-1], label="SAXS curve")
    plt.xlabel('q($\\AA^{-1}$)')
    plt.ylabel('S(q)')
    plt.title('Structure Factor Extracted from SALR Simulation')
    plt.axvspan(0.002, 0.02, color='orange', alpha=0.3, label='Aggregation Range')
    plt.axvspan(0.03, 0.09, color='green', alpha=0.3, label='Intra-cluster Correlation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"[{label}] Device: {device}, time: {t1 - t0:.6f} s, output: {out_path}")
    print(f"[{label}] end")


def main():
    print("CUDA available:", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    print("device count:", torch.cuda.device_count())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("")

    ctx = mp.get_context("spawn")
    procs = []

    cpu_proc = ctx.Process(target=_run_device, args=("cpu", "cpu"))
    procs.append(cpu_proc)

    if torch.cuda.is_available():
        gpu_proc = ctx.Process(target=_run_device, args=("gpu", "cuda"))
        procs.append(gpu_proc)
    else:
        print("CUDA not available; running CPU only.")

    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()
