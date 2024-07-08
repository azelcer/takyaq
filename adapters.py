# -*- coding: utf-8 -*-
"""
Adapters
"""


class InMicrons:
    """Wrapper for piezoelectric motors in microns."""

    _get_position = None
    _set_position = None

    def __init__(self, get_pos_func, set_pos_func):
        self._get_position = get_pos_func
        self._set_position = set_pos_func

    def get_positions(self):
        """Return positions in nm from function in micrometers."""
        return tuple(_ * 1000. for _ in self._get_position())

    def set_positions(self, x: float, y: float, z: float):
        """Set positions."""
        self._set_position(x/1000., y/1000., z/1000.)
        return
