# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:55:43 2024

The module shows how to implement an object that decides how to react after a
localization event

@author: azelcer
"""


print("************** meter  tiempo para integrar *************")
import numpy as _np
import logging as _lgn
from typing import Optional as _Optional
_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


class BaseReactor:
    """Simple Reactor. Basically a proportional response."""

    def reset_xy(self, n_xy_rois: int):
        """Initialize all neccesary internal structures.

        This simple example does not need to perform any kind of initialization
        """
        pass

    def reset_z(self):
        """Initialize all neccesary internal structures.

        This simple example does not need to perform any kind of initialization
        """
        pass

    def response(self, xy_shifts: _Optional[_np.ndarray], z_shift: float):
        """Process a mesaurement of the displacements.

        Any parameter can be NAN, so we have to take it into account.

        If xy_shifts has not been measured, a None will be received.

        Must return a 3-item tuple representing the response in x, y and z
        """
        if xy_shifts is None:
            x_shift = y_shift = 0.0
        else:
            x_shift, y_shift = _np.nanmean(xy_shifts, axis=0) * .5
        if x_shift is _np.nan:
            _lgr.warning("x shift is NAN")
            x_shift = 0.0
        if y_shift is _np.nan:
            _lgr.warning("y shift is NAN")
            y_shift = 0.0

        return -x_shift, -y_shift, z_shift


class PIReactor:
    """PI Reactor. Proportional Integral."""

    _Kp = _np.ones((3,))
    _Ki = _np.ones((3,))*.5
    _cum = _np.zeros((3,))

    def reset_xy(self, n_xy_rois: int):
        """Initialize all neccesary internal structures."""
        self._cum[0:2] = 0.
        pass

    def reset_z(self):
        """Initialize all neccesary internal structures."""
        self._cum[2] = 0.
        pass

    def response(self, xy_shifts: _Optional[_np.ndarray], z_shift: float):
        """Process a mesaurement of the displacements.

        Any parameter can be NAN, so we have to take it into account.

        If xy_shifts has not been measured, a None will be received.

        Must return a 3-item tuple representing the response in x, y and z
        """
        if xy_shifts is None:
            x_shift = y_shift = 0.0
        else:
            x_shift, y_shift = _np.nanmean(xy_shifts, axis=0) * .5
        if x_shift is _np.nan:
            _lgr.warning("x shift is NAN")
            x_shift = 0.0
        if y_shift is _np.nan:
            _lgr.warning("y shift is NAN")
            y_shift = 0.0

        error = _np.array((x_shift, y_shift, z_shift))
        self._cum += error
        rv = error * self._Kp + self._Ki * self._cum
        return -x_shift, -y_shift, z_shift