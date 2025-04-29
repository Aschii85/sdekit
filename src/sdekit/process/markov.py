
from __future__ import annotations

import re
from typing import Callable, Sequence, Self

import numpy as np
import scipy.stats

from sdekit.filtration import Filtration
from sdekit.incrementor import INCREMENTORS, Incrementor

_FUNCTIONS = {
    "sqrt": np.sqrt,
    "log": np.log,
    "cos": np.cos,
    "sin": np.sin,
}


class MarkovProcess:
    _allowed_infinitesimals = tuple(INCREMENTORS.keys())
    _instances: list[str] = []

    def __init__(self,
        name: str,
        *,
        coefficients: Sequence[MarkovProcess | Callable[[Filtration, float], np.ndarray]],
        incrementors: Sequence[Incrementor],
    ):
        self.name = name
        self.coefficients = tuple(coefficients)
        self.incrementors = tuple(incrementors)
        if len(self.coefficients) != len(self.incrementors):
            raise RuntimeError

    def __new__(cls, name: str, **kwargs) -> Self:
        if name in cls._instances:
            raise RuntimeError('Cannot create multiple processes with the same name!')
        if name in ['state', 'X', 't', 'time'] + list(_FUNCTIONS.keys()):
            raise RuntimeError('Name is protected!')
        return super().__new__(cls) # name needed?

    @classmethod
    def from_equation(cls, name: str, equation: str) -> Self:
        """Example equation equation = 'dX = (...) * dt + (...) * dWt' """
        matches = re.findall(r'\((.*?)\) \* (dt|dWt)', equation)
        coefficients = []
        incrementors = []
        for match in matches:
            if match[1] not in cls._allowed_infinitesimals:
                raise RuntimeError
            try:
                incrementors.append(INCREMENTORS[match[1]](process_name=name))
            except KeyError:
                raise NotImplementedError
            coefficients.append(_create_evaluation_function(name, match[0]))
        instance = MarkovProcess.__new__(
            cls,
            name,
        )
        MarkovProcess.__init__(
            instance,
            name,
            coefficients=coefficients,
            incrementors=incrementors
        )
        return instance

    def dist(self, start_value: float, time: float) -> scipy.stats.rv_frozen:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            return self is other
        if isinstance(other, str):
            return self.name == other
        return False


def _create_evaluation_function(name: str, equation_string: str) -> Callable[[Filtration, float], np.ndarray]:
    """
    Converts a mathematical equation string to a Python function that can be
    evaluated later, taking a state object (with a values method) and time
    as input.

    Args:
        equation_string (str): The mathematical equation as a string.
            Supports basic arithmetic (+, -, *, /), powers (^), and sqrt().
            Variables should be single letters or simple names.

    Returns:
        callable: A function that takes `state` and `time` as arguments and
                  returns the result of evaluating the equation using the
                  provided state values and time (if used in the equation).
                  Returns None if there's an error in parsing.
    """
    def replace_variable(match):
        var = match.group(0)
        if var in _FUNCTIONS.keys():
            return var
        elif var in ('t', 'time'):
            return 'time'
        elif var == 'X':
            return f'state.values(process="{name}", time=time)'
        else:
            return f'state.values(process="{var}", time=time)'

    substituted_equation = re.sub(r'[a-zA-Z_][a-zA-Z0-9_]*', replace_variable, equation_string)
    substituted_equation = substituted_equation.replace('^', '**')

    def evaluate(state: Filtration, time: float) -> np.ndarray:
        return eval(substituted_equation, {"state": state, "time": time} | _FUNCTIONS)

    return evaluate
