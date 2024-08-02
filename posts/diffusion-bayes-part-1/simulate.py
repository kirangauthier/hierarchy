# simulate.py
import numpy as np

class Simulate:
    """
    A class for simulating diffusion processes.

    This class provides methods to generate trajectories of particles
    undergoing Brownian motion. It is part of the diffusionBayes project.

    Methods:
        generate_brownianMotion: Generates a single isotropic 2D particle trajectory.
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_brownianMotion(D, n_steps, X_start, tau, base_seed):
      """
      Generate a single isotropic 2D particle trajectory.

      Parameters:
      - D (float): Diffusion coefficient [px^2 / frame]
      - n_steps (int): Number of steps in the trajectory [frame]
      - X_start (tuple): Position at time 0 (x, y) [px, px]
      - tau (float): Time step [frame]
      - base_seed (int): Random seed for this trajectory

      Returns:
      - x, y: Arrays containing the x and y positions of the trajectory [px]

      Note:
      This function uses a multivariate Gaussian step distribution to simulate Brownian motion.
      The covariance matrix reflects independence between the x and y dimensions.
      """

      np.random.seed(base_seed)
      x = np.zeros(n_steps)
      y = np.zeros(n_steps)

      x[0], y[0] = X_start

      x_traj = np.random.normal(loc=0, scale=2*D*tau, size=x.shape[0]-1)
      y_traj = np.random.normal(loc=0, scale=2*D*tau, size=x.shape[0]-1)

      x[1:] = x[0] + np.cumsum(x_traj)
      y[1:] = y[0] + np.cumsum(y_traj)

      return x, y
