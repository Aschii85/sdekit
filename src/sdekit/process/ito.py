
from __future__ import annotations

from typing import Callable

import numpy as np

from sdekit.filtration import Filtration
from sdekit.incrementor import INCREMENTORS
from sdekit.process.markov import MarkovProcess

class ItoProcess(MarkovProcess):
    _allowed_infinitesimals = ('dt', 'dWt')

    def __init__(
        self,
        name: str,
        *,
        mu: MarkovProcess | Callable[[Filtration, float], np.ndarray],
        sigma: MarkovProcess | Callable[[Filtration, float], np.ndarray],
    ):
        super().__init__(
            name=name,
            coefficients=(mu, sigma),
            incrementors=(
                INCREMENTORS['dt'](process_name=name),
                INCREMENTORS['dWt'](process_name=name),
            ),
        )
