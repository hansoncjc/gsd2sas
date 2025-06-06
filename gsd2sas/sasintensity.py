import numpy as np
from abc import ABC, abstractmethod
from structurefactor import StructureFactor
from formfactor import Sphere
from utils import _binary_mixture_intensity
import matplotlib.pyplot as plt

class Intensity(ABC):
    def __init__(self, volume_fraction, sld_sample, sld_solvent, types = None):
        self.volume_fraction = volume_fraction
        self.sld_sample = sld_sample
        self.sld_solvent = sld_solvent
        self.prefactor = self._compute_prefactor()
        self.isPoly = types is not None
        self.types = types
        self.structure_factor = None
        self.form_factor = None

    def _compute_prefactor(self):
        delta_rho = self.sld_sample - self.sld_solvent
        return self.volume_fraction * delta_rho**2

    def set_structure_factor(self, gsd_path, N_grid, types = None, frames='all'):
        self.structure_factor = StructureFactor(gsd_path, N_grid, types, frames)

    @abstractmethod
    def set_form_factor(self, *args, **kwargs):
        """Set self.form_factor in subclass."""
        pass

    @abstractmethod
    def compute_Iq(self):
        """Compute the intensity I(q) based on structure and form factors."""
        pass
    
    def plot_Iq(self, q_unit = "angstrom"):
        """Plot the computed intensity I(q) with appropriate labels."""
        
        if not hasattr(self, 'Iq'):
            self.compute_Iq()
        
        if q_unit == "angstrom":
            q = self.qr / (self.radius * 10)
            u_label = '$q(\AA^{-1})$'
        elif q_unit == "nm":
            q = self.qr / self.radius
            u_label = '$q(nm^{-1}$)'
        else:
            raise ValueError("Unsupported q_unit. Use 'angstrom' or 'nm'.")
        
        plt.figure(figsize=(8, 6), dpi=100)
        plt.loglog(q[3:-1], self.Iq[3:-1], label=f'Intensity I(q)')
        plt.xlabel(u_label, fontsize=14)
        plt.ylabel('Intensity(a.u.)', fontsize=14)



class SphereIntensity(Intensity):
    def set_form_factor(self, radius):
        """Set the form factor for a sphere. radius is in nm."""
        self.radius = np.array(radius, dtype=float)
        if self.radius.ndim > 0:
            self.isPoly = True
        else:
            self.isPoly = False
        self.form_factor = Sphere(radius)

    def compute_Iq(self):
        """Compute the intensity I(q) for spherical particles."""
        if self.isPoly:
            if self.structure_factor is None or self.form_factor is None:
                raise ValueError("Both structure and form factors must be initialized before computing I(q).")
            if len(self.radius) != 2:
                raise NotImplementedError("Only 1 or 2 radii supported for now.")
            q, Sq = self.structure_factor.compute_s_1d()
            Pq = self.form_factor.Compute_Pq(q)
            Iq = self.prefactor * Sq * Pq
            self.Iq = Iq
            self.q = q
            return q, Iq
        else:
            if self.structure_factor is None or self.form_factor is None:
                raise ValueError("Both structure and form factors must be initialized before computing I(q).")
            q, S11, S22, S12 = self.structure_factor.compute_partial_s_1d()
            P = self.form_factor.Compute_Pq(q)
            P1, P2 = P.T        # unpack columns

            # 3. mix them
            I_mix = _binary_mixture_intensity(q, S11, S22, S12, P1, P2, self.types)

            self.Iq = self.prefactor * I_mix
            self.q  = q
            return q, self.Iq