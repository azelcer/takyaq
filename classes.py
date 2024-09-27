# -*- coding: utf-8 -*-
"""
Estabilizador

clases 'utiles'
"""

import numpy as _np
from dataclasses import dataclass as _dataclass
from typing import Union


@_dataclass
class ROI:
    """Represents a ROI in pixels."""

    min_x: int
    max_x: int
    min_y: int
    max_y: int


@_dataclass
class PointInfo:
    time: float
    image: _np.ndarray
    z_shift: Union[float, None]
    xy_shifts: Union[_np.ndarray, None]
