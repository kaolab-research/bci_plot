
import numpy as np

class KernelSmoother():
    def __init__(self):
        return
    def __call__(self, x):
        return x

class GaussianKernelSmoother(KernelSmoother):
    def __init__(self, x, y, sigma: float):
        '''
        pts: shape (m,)
        sigma: float, standard deviation
        '''
        super().__init__()
        self.x = x
        self.y = y
        self.sigma = sigma
        return
    def __call__(self, xi: np.ndarray):
        '''
        x: shape (n,)
        '''
        weights = np.exp(-0.5/(self.sigma**2)*(xi[:, None] - self.x[None, :])**2)
        sums = weights@self.y
        out = sums/weights.sum(1)
        return out
    