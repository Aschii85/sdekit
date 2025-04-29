from __future__ import annotations

import numpy as np
import scipy.stats

from sdekit.process.ito import ItoProcess

class GBMProcess(ItoProcess):
    """Geometric Brownian Motion."""

    def __init__(
        self,
        name: str,
        *,
        mu: float,
        sigma: float,
    ):
        self._mu = mu
        self._sigma = sigma
        super().__init__(
            name,
            mu=lambda f, t: self._mu * f.values(time=t, process=self),
            sigma=lambda f, t: self._sigma * f.values(time=t, process=self),
        )

    def dist(self, start_value: float, time: float) -> scipy.stats.rv_frozen:
        return scipy.stats.lognorm(
            loc=np.log(start_value) + (self._mu - (self._sigma**2) / 2) * time,
            scale=self._sigma * np.sqrt(time),
        )

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def sigma(self) -> float:
        return self._sigma

class ABMProcess(GBMProcess):
    """Arithmetic Brownian Motion."""

    def __init__(
        self,
        name: str,
        *,
        mu: float,
        sigma: float,
    ):
        super().__init__(
            name,
            mu=(mu - sigma**2 / 2),
            sigma=sigma,
        )

    @property
    def mu(self) -> float:
        return self._mu + self._sigma**2/2
