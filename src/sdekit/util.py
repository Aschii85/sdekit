from __future__ import annotations

from typing import Callable, Generic, TypeVar, Type

_T = TypeVar('_T')

class classproperty(Generic[_T]):

    def __init__(self, func: Callable[[Type], _T]):
        self.fget = func

    def __get__(self, instance: object | None, owner: Type) -> _T:
        return self.fget(owner)
