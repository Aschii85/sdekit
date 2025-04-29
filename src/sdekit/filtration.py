
from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sdekit.process import MarkovProcess

class Filtration:
    def __init__(
        self,
        times: Sequence[float],
        processes: Sequence[MarkovProcess],
        values: np.ndarray, # 3d: (time, process, scenario),
        rng: np.random.Generator | None = None,
    ):
        self.processes = processes
        self.times = list(times)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._values = values
        if len(self.processes) != self._values.shape[1]:
            raise RuntimeError
        if len(self.times) != self._values.shape[0]:
            raise RuntimeError

    def values(self,
        *,
        time: float | None = None,
        process: MarkovProcess | None = None,
        scenario: int | None = None,
    ) -> np.ndarray:
        time_idx = slice(None) if time is None else self.times.index(time)
        process_idx = slice(None) if process is None else self.processes.index(process)
        scenario_idx = slice(None) if scenario is None else scenario
        return self._values[time_idx, process_idx, scenario_idx]

    def expand_values(self, time: float, values: np.ndarray)-> None:
        if time not in self.times:
            self.times.append(time)
            self._values = np.append(self._values, values, axis=0)
        else:
            self._values[self.times.index(time)] = values

    def to_dataframe(self) -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            (self.times, [p.name for p in self.processes], self.scenarios),
            names = ('time', 'process', 'scenario')
        )
        df = pd.Series(self.values().flatten(), index=index)
        return df.unstack('process')

    @property
    def scenarios(self) -> list[int]:
        return list(range(self._values.shape[2]))
