# -*- coding: utf-8 -*-
"""
Useful data interchange classes


Copyright (C) 2025 Andrés Zelcer and others

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
init_callback_type = _Callable[[StabilizationType], bool]
end_callback_type = _Callable[[StabilizationType], None]
