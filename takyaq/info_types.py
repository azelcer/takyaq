# -*- coding: utf-8 -*-
"""
Estabilizador

clases 'utiles'
"""

import numpy as _np
from dataclasses import dataclass as _dataclass
from typing import Union as _Union, Tuple as _Tuple, Callable as _Callable
from enum import Enum as _Enum


class StabilizationType(_Enum):
    XY_stabilization = 0
    Z_stabilization = 1


@_dataclass
class ROI:
    """Represents a ROI in pixels."""

    min_x: int
    max_x: int
    min_y: int
    max_y: int

    @classmethod
    def from_position_and_size(cls, position: _Tuple[float, float],
                               size: _Tuple[float, float]):
        """Create a ROI from position and size."""
        x, y = position
        w, h = size
        return cls(x, x + w, y, y + h)

    @classmethod
    def from_pyqtgraph(cls, pyqtgrapgROI):
        """Create a ROI from a PyQtGraph ROI."""
        return cls.from_position_and_size(pyqtgrapgROI.pos(), pyqtgrapgROI.size())


@_dataclass
class PointInfo:
    """Holds data for a single point in timeline."""

    time: float
    image: _np.ndarray
    z_shift: _Union[float, None]
    xy_shifts: _Union[_np.ndarray, None]


@_dataclass
class CameraInfo:
    """Holds Information about camera parameters."""

    nm_ppx_xy: float
    nm_ppx_z: float
    angle: float


report_callback_type = _Callable[[PointInfo], None]
init_callback_type = _Callable[StabilizationType, bool]
end_callback_type = _Callable[StabilizationType, None]
