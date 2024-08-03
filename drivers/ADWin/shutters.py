# -*- coding: utf-8 -*-
import time.sleep as _sleep
from ADWin import _adw, Processes


_SHUTTER_NUM_PAR = 73
_SHUTTER_STATUS_PAR = 72


def set_shutter(shutter: int, status: bool):
    """Set single shutter status"""
    _adw.Set_Par(_SHUTTER_NUM_PAR, shutter)
    setting = 1 if status else 0
    _adw.Set_Par(_SHUTTER_STATUS_PAR, setting)
    _adw.Start_Process(Processes.SHUTTERS)


def set_shutters(shutters: list, status: bool):
    """Change many shutters at once."""
    for shutter in shutters:
        set_shutter(shutter, status)
        while _adw.ProcessStatus(Processes.SHUTTERS):
            _sleep(.10)
