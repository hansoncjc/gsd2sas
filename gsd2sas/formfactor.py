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
    
class PolydisperseSphere(FormFactor):
    def __init__(self, mean_radius, std_dev, n_samples=50, distribution='normal'):
        super().__init__('polydisperse_sphere')
        self.mean_radius = mean_radius
        self.std_dev = std_dev
        self.n_samples = n_samples
        self.distribution = distribution
        self._generate_radii()

    def whatshape(self):
        print(self.shape)
        return self.shape

    def _generate_radii(self):
        if self.distribution == 'normal':
            self.radii = np.random.normal(self.mean_radius, self.std_dev, self.n_samples)
        else:
            raise ValueError("Only 'normal' distribution is currently supported.")

        self.radii = self.radii[self.radii > 0]  # Discard negative radii if any
        self.weights = np.ones_like(self.radii) / len(self.radii)

    def Compute_Fq(self, q):
        q = np.asarray(q)
        q = q[:, None]  # shape (n_q, 1)
        radii = self.radii[None, :]  # shape (1, n_radii)
        qr = q * radii               # shape (n_q, n_radii)

        with np.errstate(divide='ignore', invalid='ignore'):
            Fqs = 3 * (np.sin(qr) - qr * np.cos(qr)) / (qr**3)

        # Correct handling of q=0 rows
        zero_q_rows = np.isclose(q.flatten(), 0.0)
        Fqs[zero_q_rows, :] = 1.0

        Fq_avg = np.average(Fqs, axis=1, weights=self.weights)
        return Fq_avg
