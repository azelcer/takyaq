# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:55:43 2024

The module shows how to implement an object that decides how to react after a
localization event

@author: azelcer
"""

import numpy as _np
import logging as _lgn
from typing import Optional as _Optional, Union as _Union
from collections.abc import Collection as _Collection
from numbers import Number as _Number

_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


class BaseReactor:
    """Simple Reactor. Basically a proportional response."""

    _multiplier = 1.0

    def __init__(self, multiplier: float = 1.):
        self._multiplier = multiplier

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

    def response(self, t: float, xy_shifts: _Optional[_np.ndarray], z_shift: float):
        """Process a mesaurement of the displacements.

        Any parameter can be NAN, so we have to take it into account.

        If xy_shifts has not been measured, a None will be received.

        Must return a 3-item tuple representing the response in x, y and z
        """
        if xy_shifts is None:
            x_shift = y_shift = 0.0
        else:
            x_shift, y_shift = _np.nanmean(xy_shifts, axis=0)
        if x_shift is _np.nan:
            _lgr.warning("x shift is NAN")
            x_shift = 0.0
        if y_shift is _np.nan:
            _lgr.warning("y shift is NAN")
            y_shift = 0.0

        return (-x_shift * self._multiplier, -y_shift * self._multiplier,
                z_shift * self._multiplier)


class PIReactor:
    """PI Reactor. Proportional Integral."""

    _Kp = _np.ones((3,))
    _Ki = _np.ones((3,))
    _cum = _np.zeros((3,))
    _invert = _np.array([-1, -1, 1])
    lasttime = 0

    def __init__(self, Kp: _Union[float, _Collection[float]] = 1.,
                 Ki: _Union[float, _Collection[float]] = 1.):
        if isinstance(Kp, _Number):
            self._Kp[:] = float(Kp)
        elif len(Kp) == 3:
            self._Kp[:] = [float(_) for _ in Kp]
        else:
            raise TypeError(f"Invalid parameter used as Kp: {Kp}")
        if isinstance(Ki, _Number):
            self._Ki[:] = float(Kp)
        elif len(Ki) == 3:
            self._Ki[:] = [float(_) for _ in Ki]
        else:
            raise TypeError(f"Invalid parameter used as Ki: {Ki}")

    def reset_xy(self, n_xy_rois: int):
        """Initialize all neccesary internal structures."""
        self._cum[0:2] = 0.
        self.lasttime = 0
        # TODO: See how to manage lasttime z and XY

    def reset_z(self):
        """Initialize all neccesary internal structures."""
        self._cum[2] = 0.
        self.lasttime = 0

    def response(self, t: float, xy_shifts: _Optional[_np.ndarray], z_shift: float):
        """Process a mesaurement of the displacements.

        Any parameter can be NAN, so we have to take it into account.

        If xy_shifts has not been measured, a None will be received.

        Must return a 3-item tuple representing the response in x, y and z
        """
        if xy_shifts is None:
            x_shift = y_shift = 0.0
        else:
            x_shift, y_shift = _np.nanmean(xy_shifts, axis=0)
            if x_shift is _np.nan:
                _lgr.warning("x shift is NAN")
                x_shift = 0.0
            if y_shift is _np.nan:
                _lgr.warning("y shift is NAN")
                y_shift = 0.0

        error = _np.array((x_shift, y_shift, z_shift))
        if not self.lasttime:
            self.lasttime = t
        self._cum += error * (t - self.lasttime)
        self.lasttime = t
        rv = error * self._Kp + self._Ki * self._cum
        return rv * self._invert


class PIDReactor:
    """PID Reactor. mean of last 5 derivatives used as derivative param"""

    _Kp = _np.ones((3,)) * .85
    _Ki = _np.ones((3,)) * .30
    _Kd = _np.ones((3,)) * .15
    _deriv = _np.zeros((3,))
    _last_e = _np.zeros((3,))
    _cum = _np.zeros((3,))
    _N_VALS = 5
    next_val = 0
    _last_deriv = _np.full((_N_VALS, 3,), _np.nan)
    _invert = _np.array([-1, -1, 1])
    lasttime = 0.

    def reset_xy(self, n_xy_rois: int):
        """Initialize all neccesary internal structures."""
        self._cum[0:2] = 0.
        pass

    def reset_z(self):
        """Initialize all neccesary internal structures."""
        self._cum[2] = 0.
        pass

    def response(self, t: float, xy_shifts: _Optional[_np.ndarray], z_shift: float):
        """Process a mesaurement of the displacements.

        Any parameter can be NAN, so we have to take it into account.

        If xy_shifts has not been measured, a None will be received.

        Must return a 3-item tuple representing the response in x, y and z
        """
        if xy_shifts is None:
            x_shift = y_shift = 0.0
        else:
            x_shift, y_shift = _np.nanmean(xy_shifts, axis=0)
        if x_shift is _np.nan:
            _lgr.warning("x shift is NAN")
            x_shift = 0.0
        if y_shift is _np.nan:
            _lgr.warning("y shift is NAN")
            y_shift = 0.0

        if not self.lasttime:
            self.lasttime = t
        error = _np.array((x_shift, y_shift, z_shift))
        self._cum += error * (t - self.lasttime)
        d = error - self._last_e
        self._last_deriv[self.next_val] = d
        self.next_val = (self.next_val + 1) % self._N_VALS
        self._deriv = _np.nansum(self._last_deriv, axis=0) / self._N_VALS
        rv = error * self._Kp + self._Ki * self._cum + self._Kd * self._deriv
        self._last_e = error
        self.lasttime = t
        return rv * self._invert


class PIDReactor2:
    """Modified PID Reactor. Proportional, short-time Integral, Derivative."""

    _Kp = _np.ones((3,)) * .85
    _Ki = _np.ones((3,)) * .30
    _Kd = _np.ones((3,)) * .15
    _deriv = _np.zeros((3,))
    _last_e = _np.zeros((3,))
    _cum = _np.zeros((3,))
    _fact = 0.5
    lasttime = 0.

    def reset_xy(self, n_xy_rois: int):
        """Initialize all neccesary internal structures."""
        self._cum[0:2] = 0.
        pass

    def reset_z(self):
        """Initialize all neccesary internal structures."""
        self._cum[2] = 0.
        pass

    def response(self, t: float, xy_shifts: _Optional[_np.ndarray], z_shift: float):
        """Process a mesaurement of the displacements.

        Any parameter can be NAN, so we have to take it into account.

        If xy_shifts has not been measured, a None will be received.

        Must return a 3-item tuple representing the response in x, y and z
        """
        if xy_shifts is None:
            x_shift = y_shift = 0.0
        else:
            x_shift, y_shift = _np.nanmean(xy_shifts, axis=0)
        if x_shift is _np.nan:
            _lgr.warning("x shift is NAN")
            x_shift = 0.0
        if y_shift is _np.nan:
            _lgr.warning("y shift is NAN")
            y_shift = 0.0

        if not self.lasttime:
            self.lasttime = t
        error = _np.array((x_shift, y_shift, z_shift))
        self._cum = self._cum * self._fact + error * (t - self.lasttime)
        d = error - self._last_e
        self._deriv = self._deriv * self._fact + d
        rv = error * self._Kp + self._Ki * self._cum + self._Kd * self._deriv
        self._last_e = error
        self.lasttime = t
        return -rv
