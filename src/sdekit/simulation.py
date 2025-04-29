from collections.abc import Sequence
from typing import Literal

import numpy as np

from sdekit.filtration import Filtration
from sdekit.incrementor import WienerIncrementor, TimeIncrementor, INCREMENTORS
from sdekit.process import MarkovProcess
from sdekit.process.ito import ItoProcess

def simulate(
    *,
    method: Literal['euler', 'milstein'] = 'euler',
    processes: Sequence[MarkovProcess],
    start_values: Sequence[float], # Make into special class with labels, since confusion may arise?
    time_steps: Sequence[float],
    scenarios: int = 1_000,
    wiener_correlations: np.ndarray | None = None, # Make into special class with labels, since confusion may arise?
) -> Filtration:
    """Simulate scenarios using the Euler-Maruyama or Milstein Schemes scheme."""
    if method == 'milstein' and not all(isinstance(p, ItoProcess) for p in processes):
        raise RuntimeError
    if len(processes) != len(start_values):
        raise RuntimeError
    if wiener_correlations is not None:
        WienerIncrementor.set_correlations(wiener_correlations)
    for incremtor in INCREMENTORS.values():
        incremtor.set_num_scenarios(scenarios)
    time_steps = [t for t in time_steps if t != 0]
    start_vals = np.full((1 + len(time_steps), len(processes), scenarios), np.nan)
    for idx, val in enumerate(start_values):
        start_vals[0, idx,:] = val
    state = Filtration([0] + time_steps, processes, start_vals)
    for prev, next in zip([0] + time_steps[:-1], time_steps, strict=True):
        if method == 'milstein':
            next_values = np.array(
                    [
                        _next_scenario_values_milstein(processes, state, prev, next)
                    ]
                )
        else:
            next_values = np.array(
                    [
                        [
                            _next_scenario_values_euler(p, state, prev, next)
                            for p in processes
                        ]
                    ]
                )
        state.expand_values(next, next_values)
    return state

def _next_scenario_values_euler(
    process: MarkovProcess,
    state: Filtration,
    t0: float,
    t1: float,
) -> np.ndarray:
    """Calculate next scenario values using the Euler-Maruyama scheme."""
    res = state.values(time=t0, process=process)
    for coef, inc in zip(process.coefficients, process.incrementors):
        if isinstance(coef, MarkovProcess):
            c_eval = _next_scenario_values_euler(coef, state, t0, t1)
        else:
            c_eval = coef(state, t0)
        res += c_eval * inc.scenarios(t0, t1)
    return res

def _next_scenario_values_milstein(
    processes: Sequence[ItoProcess],
    state: Filtration,
    t0: float,
    t1: float,
) -> np.ndarray:
    dt = t1 - t0
    dW = np.zeros((len(processes), WienerIncrementor._num_scenarios))
    for idx, process in enumerate(processes):
        for inc in process.incrementors:
            if type(inc) == WienerIncrementor:
                dW[idx, :] = inc.scenarios(t0, t1)
    k1 = np.zeros((len(processes), WienerIncrementor._num_scenarios))
    for idx, process in enumerate(processes):
        for coef, inc in zip(process.coefficients, process.incrementors):
            if type(inc) == TimeIncrementor:
                k1[idx, :] += dt * coef(state, t0)
            if type(inc) == WienerIncrementor:
                k1[idx, :] += (dW[idx, :] - np.random.choice([-1, 1]) * np.sqrt(dt)) * coef(state, t0)
    midstate = Filtration(
        times=[t0],
        processes=state.processes,
        values=np.array([state.values(time=t0) + k1])
    )
    k2 = np.zeros((len(processes), WienerIncrementor._num_scenarios))
    for idx, process in enumerate(processes):
        for coef, inc in zip(process.coefficients, process.incrementors):
            if type(inc) == TimeIncrementor:
                k2[idx, :] += dt * coef(midstate, t0)
            if type(inc) == WienerIncrementor:
                k2[idx, :] += (dW[idx, :] + np.random.choice([-1, 1]) * np.sqrt(dt)) * coef(midstate, t0)
    return state.values(time=t0) + (k1 + k2) / 2
