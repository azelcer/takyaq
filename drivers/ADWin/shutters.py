# -*- coding: utf-8 -*-
import time.sleep as _sleep
from . import _adw, Processes


_SHUTTER_NUM_PAR = 73
_SHUTTER_STATUS_PAR = 72


def set_shutter(shutter: int, status: bool):
    """Set single shutter status.

    WARNING: shutters migh need some time to move.
    """
    _adw.Set_Par(_SHUTTER_NUM_PAR, shutter)
    setting = 1 if status else 0
    _adw.Set_Par(_SHUTTER_STATUS_PAR, setting)
    _adw.Start_Process(Processes.SHUTTERS.value)


def set_shutters(shutters: list, status: bool):
    """Change many shutters at once."""
    for shutter in shutters:
        set_shutter(shutter, status)
        while _adw.ProcessStatus(Processes.SHUTTERS.value):
            _sleep(.050)
