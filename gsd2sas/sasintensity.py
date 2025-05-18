import numpy as np
from abc import ABC, abstractmethod
from structurefactor import StructureFactor
from formfactor import Sphere

class Intensity(ABC):
    def __init__(self, volume_fraction, sld_sample, sld_solvent):
        self.volume_fraction = volume_fraction
        self.sld_sample = sld_sample
        self.sld_solvent = sld_solvent
        self.prefactor = self._compute_prefactor()
        
        self.structure_factor = None
        self.form_factor = None

    def _compute_prefactor(self):
        delta_rho = self.sld_sample - self.sld_solvent
        return self.volume_fraction * delta_rho**2

    def set_structure_factor(self, gsd_path, N_grid, frame='all'):
        self.structure_factor = StructureFactor(gsd_path, N_grid, frame)

    @abstractmethod
    def set_form_factor(self, *args, **kwargs):
        """Set self.form_factor in subclass."""
        pass

    def compute_Iq(self):
        if self.structure_factor is None or self.form_factor is None:
            raise ValueError("Both structure and form factors must be initialized before computing I(q).")
        q, Sq = self.structure_factor.compute_s_1d()
        Pq = self.form_factor.Pq(q)
        Iq = self.prefactor * Sq * Pq
        return q, Iq


class SphereIntensity(Intensity):
    def set_form_factor(self, radius):
        self.form_factor = Sphere(radius)
