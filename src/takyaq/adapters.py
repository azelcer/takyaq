# -*- coding: utf-8 -*-
"""
Convenience wrappers for piezo interfaces.
"""

from base_classes import BasePiezo


class InMicrons(BasePiezo):
    """Wrapper for piezoelectric stages in microns."""

    _get_position = None
    _set_position = None

    def __init__(self, get_pos_func, set_pos_func):
        """Class that wraps piezo functions that manage values in micrometers.

        Parameters
        ----------
        get_pos_func : Callable
            Function that returns a tuple of (x, y, z) positions in micrometers.
        set_pos_func : callable
            Function that takes as parameters x, y and z positions in micrometers.
        """
        self._get_position = get_pos_func
        self._set_position = set_pos_func

    def get_position(self):
        """Return positions in nm from function in micrometers."""
        return tuple(_ * 1000. for _ in self._get_position())

    def set_position(self, x: float, y: float, z: float):
        """Set positions using parameters in nanometers."""
        self._set_position(x/1000., y/1000., z/1000.)
        return


class InNanometers(BasePiezo):
    """Wrapper for piezoelectric stages."""

    _get_position = None
    _set_position = None

    def __init__(self, get_pos_func, set_pos_func):
        """Class that wraps piezo functions into a common class.

        Parameters
        ----------
        get_pos_func : Callable
            Function that returns a tuple of (x, y, z) positions in nanometers.
        set_pos_func : Callable
            Function that takes as parameters x, y and z positions in nanometers.
        """
        self._get_position = get_pos_func
        self._set_position = set_pos_func

    def get_position(self):
        """Return position in nm."""
        return self._get_position()

    def set_position(self, x: float, y: float, z: float):
        """Set position."""
        self._set_position(x, y, z)
        return
