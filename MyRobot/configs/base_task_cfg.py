from __future__ import annotations
from typing import Callable, Literal
from dataclasses import MISSING

from metasim.utils import configclass
from metasim.queries import ContactForces

@configclass
class BaseTaskCfg:
    """
    The base class of configuration for My Robot tasks.
    """
