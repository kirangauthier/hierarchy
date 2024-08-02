import numpy as np

class Infer:
    """
    A class for performing inference on diffusion processes.

    This class provides methods to infer parameters of diffusion processes
    based on observed trajectories. It is part of the diffusionBayes project.

    Methods:
        infer_diffusivity: Infers the diffusivity parameter from a given trajectory.
    """

    def __init__(self):
        pass

    @staticmethod
    def infer_diffusivity(trajectory, inference_step=1, drift=True):
        """
        Infer the diffusivity parameter from a given trajectory.

        This method computes the posterior parameters (alpha, beta) for an inverse
        gamma distribution of the diffusivity, based on the observed displacements.

        Parameters:
        trajectory (numpy.ndarray): An array of shape (N, 3) where each row represents
                                    a point in the trajectory. The columns represent
                                    time, x-coordinate, and y-coordinate respectively.
        inference_step (int): The step size for inference. Default is 1.
        drift (bool): If True, estimates and accounts for drift in the process.
                      If False, assumes no drift. Default is True.

        Returns:
        tuple: A tuple containing two float values:
               - alpha: shape parameter of the inverse gamma distribution
               - beta: scale parameter of the inverse gamma distribution

        Note:
        The returned alpha and beta can be used to characterize the posterior
        distribution of the diffusivity parameter.
        """
        x, y = trajectory
        t = np.arange(len(x))

        # compute displacements
        idx = (np.mod(t, inference_step) == 0)
        dt = t[idx][1:] - t[idx][0:-1]
        dx = x[idx][1:] - x[idx][0:-1]
        dy = y[idx][1:] - y[idx][0:-1]

        K = dx.shape[0]

        # estimate drift parameters
        if drift:
            Uhat = np.sum(dx) / np.sum(dt)
            Vhat = np.sum(dy) / np.sum(dt)

            alpha = K - 2
            beta = np.sum(((dx - Uhat*dt)**2 + (dy - Vhat*dt)**2) / (4*dt))

        # compute posterior parameters for inverse gamma distribution
        else:
            alpha = K - 1
            beta = np.sum((dx**2 + dy**2) / (4*dt))

        return alpha, beta
