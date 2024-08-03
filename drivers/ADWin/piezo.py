# -*- coding: utf-8 -*-
from ADWin import _adw, Processes as _Processes

# FPAR of actuator positions
_X_ACTUATOR_FPAR = 70
_Y_ACTUATOR_FPAR = 71
_Z_ACTUATOR_FPAR = 72

# FPArt of positions to move to.
_X_MOVE_FPAR = 23
_Y_MOVE_FPAR = 24
_Z_MOVE_FPAR = 25


def set_moveTo_param(x: float, y: float, z: float, n_pixels_x: int = 128,
                     n_pixels_y: int = 128, n_pixels_z: int = 128, pixeltime=2000):
    """Setea los parámatros para mover a una posición x,y,z."""
    x_f = tools.convert(x, 'XtoU')
    y_f = tools.convert(y, 'XtoU')
    z_f = tools.convert(z, 'XtoU')

    self.adw.Set_Par(21, n_pixels_x)
    self.adw.Set_Par(22, n_pixels_y)
    self.adw.Set_Par(23, n_pixels_z)

    _adw.Set_FPar(_X_MOVE_FPAR, x_f)
    _adw.Set_FPar(_X_MOVE_FPAR, x_f)
    _adw.Set_FPar(_X_MOVE_FPAR, x_f)

    _adw.Set_FPar(26, tools.timeToADwin(pixeltime))


def simple_move(x: float, y: float, z: float):
    # Fallar si el proceso actuador est'a corriendo
    set_moveTo_param(x, y, z)
    _Start_Process(_Processes.MOVETO_XYZ)
