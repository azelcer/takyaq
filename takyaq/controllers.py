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


class boxaverage:
    """Helper boxed average class."""

    _data: _np.ndarray
    _n_points: int
    _cur_pos: int = 0

    def __init__(self, n_points: int):
        self._n_points = n_points
        self._data = _np.full((n_points,), _np.nan)

    def reset(self):
        self._data[:] = _np.nan
        self._cur_pos = 0

    def put(self, value):
        self._data[self._cur_pos] = value
        self._cur_pos = (self._cur_pos + 1) % self._n_points

    def get(self):
        return _np.nanmean(self._data)


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
        self.set_Kp(Kp)
        self.set_Ki(Ki)

    def set_Kp(self, Kp: _Union[float, _Collection[float]]):
        self._Kp[:] = _np.array(Kp)

    def set_Ki(self, Ki: _Union[float, _Collection[float]]):
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


class SmoothedPIController:
    """PI Controller. Proportional Integral, smoothed."""

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
        self.set_Kp(Kp)
        self.set_Ki(Ki)
        self._x_boxed = boxaverage(5)
        self._y_boxed = boxaverage(5)
        self._z_boxed = boxaverage(5)
        

    def set_Kp(self, Kp: _Union[float, _Collection[float]]):
        self._Kp[:] = _np.array(Kp)

    def set_Ki(self, Ki: _Union[float, _Collection[float]]):
        self._Ki[:] = _np.array(Ki)

    def set_Kd(self, Kd: _Union[float, _Collection[float]]):
        ...

    def reset_xy(self, n_xy_rois: int):
        """Initialize all neccesary internal structures."""
        self._cum[0:2] = 0.
        self._last_times[0:2] = 0.
        self._x_boxed.reset()
        self._y_boxed.reset()

    def reset_z(self):
        """Initialize all neccesary internal structures."""
        self._cum[2] = 0.
        self._last_times[2] = 0.
        self._z_boxed.reset()

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
        self._x_boxed.put(x_shift)
        self._y_boxed.put(y_shift)
        self._z_boxed.put(z_shift)
        error = _np.array((self._x_boxed.get(), self._y_boxed.get(), self._z_boxed.get()))
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
    _last_times = _np.zeros((3,))
    _n_deriv_points = 10

    def __init__(self, Kp: _Union[float, _Collection[float]] = 1.,
                 Ki: _Union[float, _Collection[float]] = 1.,
                 Kd: _Union[float, _Collection[float]] = 1.,
                 deriv_points: int = 10):
        self.set_Kp(Kp)
        self.set_Ki(Ki)
        self.set_Kd(Kd)
        self._n_deriv_points = deriv_points
        self._last_deriv = _np.full((deriv_points, 3,), _np.nan)

    def set_Kp(self, Kp: _Union[float, _Collection[float]]):
        self._Kp[:] = _np.array(Kp)

    def set_Ki(self, Ki: _Union[float, _Collection[float]]):
        self._Ki[:] = _np.array(Ki)

    def set_Kd(self, Kd: _Union[float, _Collection[float]]):
        self._Kd[:] = _np.array(Kd)

    def reset_xy(self, n_xy_rois: int):
        """Initialize all neccesary internal structures."""
        self._cum[0:2] = 0.
        self._last_deriv[:, 0:2] = _np.nan
        self._last_times[0:2] = 0.
        self._last_e[0:2] = 0.

    def reset_z(self):
        """Initialize all neccesary internal structures."""
        self._cum[2] = 0.
        self._last_deriv[:, 2] = _np.nan
        self._last_times[2] = 0.
        self._last_e[2] = 0.

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
        # adapt delta-t to deriv
        delta_t[_np.where(delta_t <= 0)] = _np.inf
        d = (error - self._last_e) / delta_t

        self._last_deriv[self.next_val] = d
        self.next_val = (self.next_val + 1) % self._n_deriv_points
        self._deriv = _np.nanmean(self._last_deriv, axis=0)
        # print(self._deriv, self._Kd)
        rv = error * self._Kp + self._Ki * self._cum + self._Kd * self._deriv
        self._last_e = error
        self._last_times[:] = t
        return -rv



class PIController2:
    """PI Controller."""

    _Kp = _np.ones((3,))
    _Ki = _np.ones((3,))
    _deriv = _np.zeros((3,))
    _last_e = _np.zeros((3,))
    _cum = _np.zeros((3,))
    next_val = 0
    _last_times = _np.zeros((3,))

    def __init__(self, Kp: _Union[float, _Collection[float]] = 1.,
                 Ki: _Union[float, _Collection[float]] = 1.,
                 ):
        self.set_Kp(Kp)
        self.set_Ki(Ki)

    def set_Kp(self, Kp: _Union[float, _Collection[float]]):
        self._Kp[:] = _np.array(Kp)

    def set_Ki(self, Ki: _Union[float, _Collection[float]]):
        self._Ki[:] = _np.array(Ki)

    def reset_xy(self, n_xy_rois: int):
        """Initialize all necesary internal structures for XY."""
        self._cum[0:2] = 0.
        self._last_times[0:2] = 0.

    def reset_z(self):
        """Initialize all necesary internal structures for Z."""
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
        rv = error * self._Kp + self._Ki * self._cum
        self._last_times[:] = t
        return -rv
