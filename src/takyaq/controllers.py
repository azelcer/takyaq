# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:55:43 2024

The module implement objects that react after a fiduciary localization event

@author: azelcer
"""

import numpy as _np
import logging as _lgn
from typing import Optional as _Optional, Union as _Union
from collections.abc import Collection as _Collection

_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


class ProportionalController:
    """Simple Controller. Just a the same proportional response for all axes."""

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
                -z_shift * self._multiplier)


class PIController:
    """PI Controller. Proportional Integral."""

    _Kp = _np.ones((3,))
    _Ki = _np.ones((3,))
    _cum = _np.zeros((3,))
    _last_times = _np.zeros((3,))

    def __init__(self, Kp: _Union[float, _Collection[float]] = 1.,
                 Ki: _Union[float, _Collection[float]] = 1.):
        """Proportional controller.

        Parameters
        ==========
            Kp: float or collection[3]
                Proportional term constant. Single value or one for x, y, and z
            Ki: float or collection[3]
                Intergral term constant. Single value or one for x, y, and z
        """
        self._Kp[:] = _np.array(Kp)
        self._Ki[:] = _np.array(Ki)

    def reset_xy(self, n_xy_rois: int):
        """Initialize all neccesary internal structures."""
        self._cum[0:2] = 0.
        self._last_times[0:2] = 0.

    def reset_z(self):
        """Initialize all neccesary internal structures."""
        self._cum[2] = 0.
        self._last_times[2] = 0.

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
        self._last_times[_np.where(self._last_times <= 0.)] = t
        delta_t = t - self._last_times
        delta_t[_np.where(delta_t > 1)] = 1.  # protect against suspended processes
        self._cum += error * delta_t
        self._last_times[:] = t
        rv = error * self._Kp + self._Ki * self._cum
        return -rv


class PIDController:
    """PID Controller. mean of last 5 derivatives used as derivative param."""

    _Kp = _np.ones((3,))
    _Ki = _np.ones((3,))
    _Kd = _np.ones((3,))
    _deriv = _np.zeros((3,))
    _last_e = _np.zeros((3,))
    _cum = _np.zeros((3,))
    next_val = 0
    _last_deriv = _np.full((1, 3,), _np.nan)
    lasttime = 0.

    def __init__(self, Kp: _Union[float, _Collection[float]] = 1.,
                 Ki: _Union[float, _Collection[float]] = 1.,
                 Kd: _Union[float, _Collection[float]] = 1.,
                 deriv_points: int = 10):
        self._Kp[:] = _np.array(Kp)
        self._Ki[:] = _np.array(Ki)
        self._Kd[:] = _np.array(Kd)
        self._deriv_points = deriv_points
        self._last_deriv = _np.full((deriv_points, 3,), _np.nan)

    def reset_xy(self, n_xy_rois: int):
        """Initialize all neccesary internal structures."""
        self._cum[0:2] = 0.
        self._last_deriv[:, 0:2] = 0

    def reset_z(self):
        """Initialize all neccesary internal structures."""
        self._cum[2] = 0.
        self._last_deriv[:, 2] = 0.

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
        if not self.lasttime:
            self.lasttime = t
        delta_t = t - self.lasttime
        if delta_t > 1.:  # protect against suspended processes
            delta_t = 1.
        self._cum += error * delta_t
        d = (error - self._last_e) / delta_t
        self._last_deriv[self.next_val] = d
        self.next_val = (self.next_val + 1) % self._N_VALS
        self._deriv = _np.nansum(self._last_deriv, axis=0) / self._N_VALS
        rv = error * self._Kp + self._Ki * self._cum + self._Kd * self._deriv
        self._last_e = error
        self.lasttime = t
        return rv


# class ScaledReactor:
#     """Proportional/partial response for large/small distances."""

#     _multiplier = 1.0

#     def __init__(self, xylimit: float = 3., zlimit: float = 5.,
#                  multiplier: float = 1.):
#         self._multiplier = multiplier
#         self._factor = 1. / _np.array((xylimit, xylimit, zlimit), dtype=_np.float64)
#         self._buffer = _np.ones((2, 3,), dtype=_np.float64)

#     def reset_xy(self, n_xy_rois: int):
#         """Initialize all neccesary internal structures.

#         Not needed.
#         """
#         pass

#     def reset_z(self):
#         """Initialize all neccesary internal structures.

#         Not needed.
#         """
#         pass

#     def response(self, t: float, xy_shifts: _Optional[_np.ndarray], z_shift: float):
#         """Process a mesaurement of the displacements.

#         Any parameter can be NAN, so we have to take it into account.

#         If xy_shifts has not been measured, a None will be received.

#         Must return a 3-item tuple representing the response in x, y and z
#         """
#         if xy_shifts is None:
#             x_shift = y_shift = 0.0
#         else:
#             x_shift, y_shift = _np.nanmean(xy_shifts, axis=0)
#         if x_shift is _np.nan:
#             _lgr.warning("x shift is NAN")
#             x_shift = 0.0
#         if y_shift is _np.nan:
#             _lgr.warning("y shift is NAN")
#             y_shift = 0.0
#         # TODO: do not create an array each time
#         rv = _np.array((x_shift, y_shift, -z_shift,))
#         self._buffer[0, :] = _np.abs(rv)
#         self._buffer[0] *= self._factor
#         factor = self._buffer.min(axis=0)
#         return -self._multiplier * factor * rv
