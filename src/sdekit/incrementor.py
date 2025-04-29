from __future__ import annotations

import abc
import functools
from collections import defaultdict
from typing import Type

import numpy as np
import scipy.stats

from sdekit.util import classproperty

# TODO: Support for quasi-monte carlo samplers for _sampler
class Incrementor(abc.ABC):
    _num_scenarios: int = 1_000
    __rng: dict[Type[Incrementor], np.random.Generator] = defaultdict(np.random.default_rng)
    __sampler: dict[Type[Incrementor], np.random.Generator | scipy.stats.qmc.QMCEngine] = defaultdict(np.random.default_rng)
    __correlations: dict[Type[Incrementor], np.ndarray | None] = defaultdict(lambda: None)
    __process_names: dict[Type[Incrementor], list[str]] = defaultdict(list)

    def __init__(
        self,
        *,
        process_name: str,
    ):
        self.process_name = process_name
        self.__process_names[self.__class__].append(self.process_name)

    @classproperty # type: ignore
    def _rng(cls) -> np.random.Generator:
        return cls.__rng[cls] # type: ignore

    @classproperty # type: ignore
    @functools.lru_cache
    def _sampler(cls) -> np.random.Generator | scipy.stats.qmc.QMCEngine:
        return cls.__sampler[cls]  # type: ignore

    @classproperty # type: ignore
    def _process_names(cls) -> list[str]:
        return cls.__process_names[cls] # type: ignore

    @classproperty # type: ignore
    def _correlations(cls) -> np.ndarray | None:
        return cls.__correlations[cls] # type: ignore

    @classmethod
    def set_num_scenarios(cls, num_scenarios: int) -> None:
        cls._num_scenarios = num_scenarios

    @classmethod
    def set_correlations(cls, corrs: np.ndarray | None) -> None:
        if corrs is not None and (
            (len(corrs) != len(cls._process_names)) or
            (len(corrs.shape) != 2) or
            (corrs.shape[0] != corrs.shape[1])
        ):
            raise RuntimeError
        cls.__correlations[cls] = corrs

    @classmethod
    def set_rng(cls, rng: np.random.Generator) -> None:
        cls.__rng[cls] = rng

    @classmethod
    def set_sampling_method(
        cls,
        sampling_method: Type[np.random.Generator] | Type[scipy.stats.qmc.QMCEngine],
    ) -> None:
        if isinstance(sampling_method, type(scipy.stats.qmc.QMCEngine)):
            sampler = sampling_method(
                d=cls._num_scenarios,
                optimization='random-cd',
                rng=cls._rng,
            )
        elif sampling_method is np.random.Generator:
            sampler = np.random.default_rng()
        else:
            raise NotImplementedError
        cls.__sampler[cls] = sampler

    @abc.abstractmethod
    def scenarios(self, t0: float, t1: float) -> np.ndarray: ...

    @classmethod
    @functools.lru_cache(maxsize=3)
    def _correlated_scenarios(cls, t0: float, t1: float) -> np.ndarray:
        raise NotImplementedError


class TimeIncrementor(Incrementor):
    def scenarios(self, t0: float, t1: float) -> np.ndarray:
        return np.full(self._num_scenarios, t1 - t0)


class WienerIncrementor(Incrementor):
    def scenarios(self, t0: float, t1: float) -> np.ndarray:
        if self._correlations is None:
            if isinstance(self._sampler, scipy.stats.qmc.QMCEngine):
                return scipy.stats.norm(
                    loc=0,
                    scale=np.sqrt(t1 - t0),
                ).ppf(
                    self._sampler.random(1)[0],
                )
            return self._sampler.normal(
                loc=0,
                scale=np.sqrt(t1 - t0),
                size=self._num_scenarios,
            )
        return self._correlated_scenarios(t0, t1)[self._process_names.index(self.process_name)]

    @classmethod
    @functools.lru_cache(maxsize=3)
    def _correlated_scenarios(cls, t0: float, t1: float) -> np.ndarray:
        if cls._correlations is None:
            raise RuntimeError
        if isinstance(cls._sampler, scipy.stats.qmc.QMCEngine):
            uncorrelated = scipy.stats.norm(
                loc=0,
                scale=np.sqrt(t1 - t0),
            ).ppf(
                cls._sampler.random(len(cls._process_names)),
            )
        else:
            uncorrelated = cls._sampler.normal(
                loc=0,
                scale=np.sqrt(t1 - t0),
                size=(len(cls._process_names), cls._num_scenarios),
            )
        # TODO: Check if below is correct...
        L = np.linalg.cholesky(cls._correlations)
        correlated = L @ uncorrelated
        return correlated


INCREMENTORS: dict[str, type[Incrementor]] = {
    'dt': TimeIncrementor,
    'dWt': WienerIncrementor,
}
