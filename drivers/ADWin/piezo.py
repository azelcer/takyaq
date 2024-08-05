# -*- coding: utf-8 -*-
from ADWin import (_adw, Processes as _Processes, microsec_to_ADwin as _us2A,
                   um2ADwin as _um2ADwin, ADwin2um as _ADwin2um)
from enum import Enum as _Enum

# TODO: create a manager object that keeps track of notification callbacks, etc.

# Move parameters
# FPAR of current positions (updated on all scipts)
_X_CURRENT_FPAR = 70
_Y_CURRENT_FPAR = 71
_Z_CURRENT_FPAR = 72

# FPAR of actuator positions
_X_ACTUATOR_FPAR = 40
_Y_ACTUATOR_FPAR = 41
_Z_ACTUATOR_FPAR = 32

# minimum time for actuators
_XY_ACTUATOR_TIME = 46
_Z_ACTUATOR_TIME = 36

# FPAr of positions to move to in single movement.
_X_MOVE_FPAR = 23
_Y_MOVE_FPAR = 24
_Z_MOVE_FPAR = 25

# Steps for single movement
_X_STEPS_PAR = 21
_Y_STEPS_PAR = 22
_Z_STEPS_PAR = 23

_TIME_PER_PIXEL_PAR = 26

# Scanning parameters
# Parameters that selects the fast and slow scan directions
_FAST_DIR_PAR = 10
_SLOW_DIR_PAR = 11

# piezo DACs in adwin
_X_DAC = 1
_Y_DAC = 2
_Z_DAC = 6


# global flags
_Z_ACTUATOR_RUNNING = False  # equivalent to checking Processes.ACTUATOR_Z
_XY_ACTUATOR_RUNNING = False  # equivalent to checking Processes.ACTUATOR_XY


class ScanType(_Enum):
    XY = (_X_DAC, _Y_DAC)
    XZ = (_X_DAC, _Z_DAC)
    YZ = (_Y_DAC, _Z_DAC)


def _prepare_simplemove(x: float, y: float, z: float, n_pixels_x: int = 128,
                        n_pixels_y: int = 128, n_pixels_z: int = 128, pixeltime=2000):
    """Setea los parámatros para mover a una posición x,y,z.

    El movimiento se realiza en forma suave. para cada dirección se lleva a cabo
    en n_pixel pasos, de pixeltime µs cada uno. O sea que para los parámetros por
    defecto el movimiento lleva 2000 µs * 128 pasos = 256 ms
    """
    x_f = _um2ADwin(x)
    y_f = _um2ADwin(z)
    z_f = _um2ADwin(z)

    _adw.Set_Par(_X_STEPS_PAR, n_pixels_x)
    _adw.Set_Par(_Y_STEPS_PAR, n_pixels_y)
    _adw.Set_Par(_Z_STEPS_PAR, n_pixels_z)

    _adw.Set_FPar(_X_MOVE_FPAR, x_f)
    _adw.Set_FPar(_Y_MOVE_FPAR, y_f)
    _adw.Set_FPar(_Z_MOVE_FPAR, z_f)

    _adw.Set_FPar(_TIME_PER_PIXEL_PAR, _us2A(pixeltime))


def simple_move(x: float, y: float, z: float, n_pixels_x: int = 128,
                n_pixels_y: int = 128, n_pixels_z: int = 128, pixeltime=2000):
    """Se mueve sueavemente a la posición pedida"""
    if _Z_ACTUATOR_RUNNING or _XY_ACTUATOR_RUNNING:
        # _lgr.warning("No puedo moverme con el actuador andando")
        return
    _prepare_simplemove(x, y, z, n_pixels_x, n_pixels_y, n_pixels_z, pixeltime)
    _adw.Start_Process(_Processes.MOVETO_XYZ)


def get_current_position():
    return tuple(_ADwin2um((_adw.Get_FPar(par))) for par in
                 (_X_CURRENT_FPAR, _Y_CURRENT_FPAR, _Z_CURRENT_FPAR))


def select_scan_type(scantype: ScanType):
    fast, slow = scantype.value
    _adw.Set_FPar(_FAST_DIR_PAR, fast)
    _adw.Set_FPar(_SLOW_DIR_PAR, slow)


def start_z_actuator():
    global _Z_ACTUATOR_RUNNING
    if not _Z_ACTUATOR_RUNNING:
        _adw.Start_Process(_Processes.ACTUATOR_Z)
        _Z_ACTUATOR_RUNNING = True


def stop_z_actuator():
    global _Z_ACTUATOR_RUNNING
    if _Z_ACTUATOR_RUNNING:
        _adw.Stop_Process(_Processes.ACTUATOR_Z)
        _Z_ACTUATOR_RUNNING = False


def start_xy_actuator():
    global _XY_ACTUATOR_RUNNING
    if not _XY_ACTUATOR_RUNNING:
        _adw.Start_Process(_Processes.ACTUATOR_XY)
        _XY_ACTUATOR_RUNNING = True


def stop_xy_actuator():
    global _XY_ACTUATOR_RUNNING
    if _XY_ACTUATOR_RUNNING:
        _adw.Stop_Process(_Processes.ACTUATOR_XY)
        _XY_ACTUATOR_RUNNING = False


# TODO: implementar escaneos
    # self.adw.Set_Par(1, self.tot_pixels)
    # self.data_t_adwin = tools.timeToADwin(self.data_t)
    # self.data_x_adwin = tools.convert(self.data_x, 'XtoU')
    # self.data_y_adwin = tools.convert(self.data_y, 'XtoU')

    # self.adw.SetData_Long(self.data_t_adwin, 2, 1, self.time_range)
    # self.adw.SetData_Long(self.data_x_adwin, 3, 1, self.space_range)
    # self.adw.SetData_Long(self.data_y_adwin, 4, 1, self.space_range)


# def trace_acquisition(self, Npoints, pixeltime):
#     """ 
#     Method to acquire a trace of photon counts at the current position.
#     Npoints = number of points to be acquired (max = 1024)
#     pixeltime = time per point (in μs)
#     """
#     # pixeltime in μs
    # if Npoints > 1024:
    #     _lg.warning("Requested number of trace points (%s) excedes max", Npoints)
    #     Npoints = 1024

#     self.adw.Set_FPar(65, tools.timeToADwin(pixeltime))
#     self.adw.Set_Par(60, Npoints+1)

#     self.adw.Start_Process(6)
#     trace_time = Npoints * (pixeltime/1000)  # target linetime in ms
#     wait_time = trace_time * 1.05 # TO DO: optimize this, it should work with 1.00, or maybe even less?
#                                  # it should even work without the time.sleep()
#     time.sleep(wait_time/1000) # in s
#     trace_data = self.adw.GetData_Long(6, 0, Npoints+1)

#     trace_data = trace_data[1:]# TO DO: fix the high count error on first element

#     return trace_data