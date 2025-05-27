import numpy as np
from abc import ABC, abstractmethod

class FormFactor(ABC):
    """Abstract base class for particle form factors."""

    def __init__(self, shape_name):
        self.shape = shape_name  # Fixed typo

    @abstractmethod
    def whatshape(self):
        """Returns the shape name."""
        return self.shape

    @abstractmethod
    def Compute_Fq(self, q):
        """Returns the scattering amplitude F(q) for given q."""
        pass

    def Compute_Pq(self, q):
        """Returns form factor P(q) = |F(q)|^2 for given qr."""
        return np.abs(self.Compute_Fq(q))**2

class Sphere(FormFactor):
    """Class for spherical form factor."""

    def __init__(self, radius):
        super().__init__('sphere')
        self.radius = radius

    def whatshape(self):
        print(self.shape)
        return self.shape

    def Compute_Fq(self, qr):
        """Returns the scattering amplitude F(q) for a sphere, with input of nondimensional qr."""
        with np.errstate(divide='ignore', invalid='ignore'):
            Fq = 3 * (np.sin(qr) - qr * np.cos(qr)) / (qr**3)
            Fq = np.where(qr == 0, 1.0, Fq)  # Define F(0) = 1
        return Fq
