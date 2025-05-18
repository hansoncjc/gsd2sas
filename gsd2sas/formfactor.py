import numpy as np
from abc import ABC, abstractmethod

class FormFactor(ABC):
    """Abstract base class for particle form factors."""

    def __init__(self, type_name):
        self.type = type_name  # String representing the shape

    @abstractmethod
    def shape_name(self):
        """Returns the shape name."""
        return self.type
    def Fq(self, q):
        """Returns the scattering amplitude F(q) for given q."""
        pass

    def Pq(self, q):
        """Returns form factor P(q) = |F(q)|^2 for given q."""
        return np.abs(self.Fq(q))**2
    
class Sphere(FormFactor):
    """Class for spherical form factor."""

    def __init__(self, radius):
        super().__init__('sphere')
        self.radius = radius

    def shape(self):
        print(self.type)
        return self.type

    def Compute_Fq(self, q):
        """Returns the scattering amplitude F(q) for a sphere."""
        return 3 * (np.sin(q * self.radius) - q * self.radius 
                    * np.cos(q * self.radius)) / (q * self.radius)**3
    def Compute_Pq(self, q):
        """Returns the form factor P(q) for a sphere."""
        return np.abs(self.Fq(q))**2