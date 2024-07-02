#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:02:01 2024

@author: azelcer
"""

import numpy as _np


class PI(object):
    """
    Discrete PI control
    """

    def __init__(self, setpoint:_np.ndarray, multiplier=1., kp=0., ki=0.):
        """

        Parameters
        ----------
        setpoint : _np.ndarray
            DESCRIPTION.
        multiplier : TYPE, optional
            DESCRIPTION. The default is 1..
        kp : TYPE, optional
            DESCRIPTION. The default is 0..
        ki : TYPE, optional
            DESCRIPTION. The default is 0..

        Los últimos 3 parámetros podrían ser arrays

        Returns
        -------
        None.

        """
        self._kp = multiplier * kp
        self._ki = multiplier * ki
        self._setpoint = setpoint
        self.multiplier = multiplier

        self._last_error = _np.zeros_like(setpoint)
        self._last_output = _np.zeros_like(setpoint)

    def update(self, current):
        """
        Calculate PID output value for given reference input and feedback.
        I'm using the iterative formula to avoid integrative part building.
        ki, kp > 0
        """
        error = self._setpoint - current

        if self.started:
            d_error = error - self._last_error
            self._last_output = self._last_output + self.kp * d_error + self.ki * error
        else:
            # This only runs in the first step
            self._last_output = self.kp * error
            self.started = True

        self.last_error = error

        return self._last_output
