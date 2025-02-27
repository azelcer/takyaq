#!/usr/bin/env python3
"""
Abstract base classes for Takyaq.

This module is provided to ensure that at least the method names and signatures are
honoured. Please note that Takyaq uses nm as units in all calls.


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
from abc import ABC as _ABC, abstractclassmethod as _abstractclassmethod
import numpy as _np
from typing import Optional as _Optional, Tuple as _Tuple


class BaseCamera(_ABC):
    """Base class for cameras."""

    @_abstractclassmethod
    def get_image(self) -> _np.ndarray:
        """Return an image as a 2D numpy array."""
        ...


class BasePiezo(_ABC):
    """Base class for piezoelectric stages."""

    @_abstractclassmethod
    def get_position(self) -> _Tuple[float, float, float]:
        """Return (x, y, z) position of the piezo in nanometers."""
        ...

    @_abstractclassmethod
    def set_position_xy(self, x: float, y: float):
        """Move to xy position specified in nanometers."""
        ...

    @_abstractclassmethod
    def set_position_z(self, z: float):
        """Move to z position, specified in nanometers."""
        ...

    def init(self):
        """Perform initialization that must be performed on the running thread."""
        ...


class BaseController(_ABC):
    """Base class for controllers."""

    @_abstractclassmethod
    def reset_xy(self, n_xy_rois: int):
        """Initialize and reset internal structures."""
        ...

    @_abstractclassmethod
    def reset_z(self):
        """Initialize and reset internal structures."""
        ...

    @_abstractclassmethod
    def response(self, t: float, xy_shifts: _Optional[_np.ndarray],
                 z_shift: float) -> _Tuple[float, float, float]:
        """Process a mesaurement of the displacements.

        Any parameter can be NAN, so it must be properly handled

        If xy_shifts has not been measured, a None will be received.

        Must return a 3-item tuple representing the response in x, y and z
        """
        ...
