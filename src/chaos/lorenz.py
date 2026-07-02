from dataclasses import dataclass
import numpy as np


@dataclass
class LorenzParameters:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.01


class LorenzSystem:
    """
    Lorenz System — 3D chaotic system.
    The system where the butterfly effect was discovered.

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

    Classic parameters sigma=10, rho=28, beta=8/3
    Lyapunov exponent ≈ 0.906
    """

    def __init__(self, params: LorenzParameters):
        self.params = params

    def derivatives(self, x, y, z):
        p = self.params
        dx = p.sigma * (y - x)
        dy = x * (p.rho - z) - y
        dz = x * y - p.beta * z
        return dx, dy, dz

    def step(self, x, y, z):
        """One RK4 step."""
        p = self.params
        k1x, k1y, k1z = self.derivatives(x, y, z)
        k2x, k2y, k2z = self.derivatives(
            x + 0.5*p.dt*k1x,
            y + 0.5*p.dt*k1y,
            z + 0.5*p.dt*k1z
        )
        k3x, k3y, k3z = self.derivatives(
            x + 0.5*p.dt*k2x,
            y + 0.5*p.dt*k2y,
            z + 0.5*p.dt*k2z
        )
        k4x, k4y, k4z = self.derivatives(
            x + p.dt*k3x,
            y + p.dt*k3y,
            z + p.dt*k3z
        )
        x_new = x + (p.dt/6)*(k1x + 2*k2x + 2*k3x + k4x)
        y_new = y + (p.dt/6)*(k1y + 2*k2y + 2*k3y + k4y)
        z_new = z + (p.dt/6)*(k1z + 2*k2z + 2*k3z + k4z)
        return x_new, y_new, z_new

    def simulate(self, x0, y0, z0, iterations):
        tx = np.zeros(iterations)
        ty = np.zeros(iterations)
        tz = np.zeros(iterations)
        tx[0], ty[0], tz[0] = x0, y0, z0
        for i in range(1, iterations):
            tx[i], ty[i], tz[i] = self.step(
                tx[i-1], ty[i-1], tz[i-1]
            )
        return tx, ty, tz

    def compute_lyapunov(self, x0=1.0, y0=1.0, z0=1.0,
                         n_iter=50000, renorm_steps=10):
        """
        Compute largest Lyapunov exponent using
        two-trajectory separation method with
        periodic renormalization.
        """
        p = self.params
        eps = 1e-8

        x1, y1, z1 = x0, y0, z0
        x2, y2, z2 = x0 + eps, y0, z0

        lam_sum = 0.0
        count = 0

        for i in range(n_iter):
            x1, y1, z1 = self.step(x1, y1, z1)
            x2, y2, z2 = self.step(x2, y2, z2)

            if i % renorm_steps == 0:
                dx = x2 - x1
                dy = y2 - y1
                dz = z2 - z1
                d = np.sqrt(dx**2 + dy**2 + dz**2)

                if d > 0:
                    lam_sum += np.log(d / eps)
                    count += 1
                    # Renormalize
                    x2 = x1 + eps * dx / d
                    y2 = y1 + eps * dy / d
                    z2 = z1 + eps * dz / d

        if count == 0:
            return 0.0

        # Convert from per-step to per-unit-time
        return lam_sum / (count * renorm_steps * p.dt)
