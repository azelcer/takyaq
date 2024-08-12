# -*- coding: utf-8 -*-
import time.sleep as _sleep
from functools import wraps as _wraps
from . import _adw, Processes as _Processes


_SHUTTER_NUM_PAR = 73
_SHUTTER_STATUS_PAR = 72


# decorador para anadir notificaciones
def add_pos_hook(func):
    # esto o setattr
    func.__call_after = []
    func.hook_notifier = lambda x: func.__call_after.append(x)

    @_wraps(func)
    def wrapper(*args, **kwargs):
        rv = func(*args, **kwargs)
        for f in func.__call_after:
            try:
                f(*args, **kwargs)
            except Exception as e:
                # Poner logger
                print("Exception on hook: %s(%s)" % (type(e), e))
        return rv
    return wrapper


@add_pos_hook
def set_shutter(shutter: int, status: bool):
    """Set single shutter status.

    WARNING: shutters migh need some time to move.
    """
    _adw.Set_Par(_SHUTTER_NUM_PAR, shutter)
    setting = 1 if status else 0
    _adw.Set_Par(_SHUTTER_STATUS_PAR, setting)
    _adw.Start_Process(_Processes.SHUTTERS.value)


def set_shutters(shutters: list, status: bool):
    """Change many shutters at once."""
    for shutter in shutters:
        set_shutter(shutter, status)
        while _adw.ProcessStatus(_Processes.SHUTTERS.value):
            _sleep(.050)
