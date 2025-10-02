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
        self.delta_rho = self.sld_sample - self.sld_solvent
        return self.volume_fraction * self.delta_rho**2

    def set_structure_factor(self, gsd_path, N_grid, frames='all'):
        self.structure_factor = StructureFactor(gsd_path, N_grid, self.types, frames)

    @abstractmethod
    def set_form_factor(self, *args, **kwargs):
        """Set self.form_factor in subclass."""
        pass

    @abstractmethod
    def compute_Iq(self):
        """Compute the intensity I(q) based on structure and form factors."""
        pass
    
    '''
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
    '''



class SphereIntensity(Intensity):
    def set_form_factor(self, radius):
        """Set the form factor for a sphere. radius is in nm."""
        self.radius = np.array(radius, dtype=float)
        self.isPoly = (self.radius.ndim) > 0 or (self.types is not None)
        self.form_factor = Sphere(radius)

    def compute_Iq(self):
        """Compute the intensity I(q) for spherical particles."""
        if self.isPoly:
            if self.structure_factor is None or self.form_factor is None:
                raise ValueError("Both structure and form factors must be initialized before computing I(q).")
            q, S11, S22, S12 = self.structure_factor.compute_s_1d()
            F = self.form_factor.Compute_Fq(q)
            F1, F2 = F.T        # unpack columns

            I_mix = self._binary_mixture_intensity(q, S11, S22, S12, F1, F2)

            #self.Iq = self.prefactor * I_mix
            self.q  = q
            self.Iq = I_mix
            return q, I_mix
        else:
            if self.structure_factor is None or self.form_factor is None:
                raise ValueError("Both structure and form factors must be initialized before computing I(q).")
            q, Sq = self.structure_factor.compute_s_1d()
            Pq = self.form_factor.Compute_Pq(q)
            V = 4/3 * np.pi * self.radius**3
            Iq = self.prefactor * V * Sq * Pq
            self.Iq = Iq
            self.q = q
            return q, Iq

    def _binary_mixture_intensity(self, q, S11, S22, S12, F1, F2):
        """
        I(q) for a binary mixture of spheres.

        Parameters
        ----------
        q      : (Nq,)         q-grid from StructureFactor
        S11    : (Nq,)         partial S(q) for type-1–type-1
        S22    : (Nq,)         partial S(q) for type-2–type-2
        S12    : (Nq,)         partial S(q) for type-1–type-2
        F1,F2  : (Nq,)         sphere form factor amplitudes for radii R1,R2

        Returns
        -------
        I : (Nq,) ndarray
            Scattering intensity before the contrast/volume prefactor.
        """
        x2 = np.mean(self.types == self.types.max())       # mole fraction of species 2
        x1 = 1.0 - x2

        V = (4.0/3.0)*np.pi*(self.radius**3)         # (2,)
        V1, V2 = V

        phi1 = self.volume_fraction * (x1*V1) / (x1*V1 + x2*V2)
        phi2 = self.volume_fraction * (x2*V2) / (x1*V1 + x2*V2)

        a1 = phi1 * (self.delta_rho**2) * V1
        a2 = phi2 * (self.delta_rho**2) * V2

        I_mix = (a1 * S11 * (F1**2)
            + a2 * S22 * (F2**2)
            + 2.0 * np.sqrt(a1*a2) * S12 * (F1 * F2))
        return I_mix