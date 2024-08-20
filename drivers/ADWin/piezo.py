# -*- coding: utf-8 -*-
from . import (_adw, Processes as _Processes, microsec_to_ADwin as _us2A,
               ADwin as _ADwin, um2ADwin as _um2ADwin, ADwin2um as _ADwin2um)
from enum import Enum as _Enum
import numpy as _np
import ctypes as _ct
import time as _time
import logging as _lgn
from dataclasses import dataclass as _dataclass

_lgr = _lgn.getLogger(__name__)

# TODO: create a manager object that keeps track of notification callbacks, etc.

# Move parameters
# FPAR of current positions (updated on all ADwin scripts, but check scan_line)
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

# DATA ARRAYS
_OUTPUT_ARRAY = 1
_SCAN_TIMES_ARRAY = 2
_FAST_DIR_POSITIONS_ARRAY = 3
_SLOW_DIR_POSITIONS_ARRAY = 4

# MISC
_LINE_LENGTH_PAR = 1

# Increment in slow direction
_SLOW_STEP_FSCAN = 2

# piezo DACs in adwin
_X_DAC = 1
_Y_DAC = 2
_Z_DAC = 6


class ScanType(_Enum):
    XY = (_X_DAC, _Y_DAC)
    XZ = (_X_DAC, _Z_DAC)
    YZ = (_Y_DAC, _Z_DAC)


@_dataclass
class ScanInfo:
    """Intercambia (entrada/salida) informacion sobre un scan."""
    scan_range: float  # scan range in um
    n_pixels: int  # number of pixels on a scan side
    t_pixel: float  # time per pixel in us
    n_aux_pixels: int  # Numero de pixeles auxiliares para acelerar
    a_aux: list  # aceleracion en cada segmento)
    x_initial: float  # Posiciones iniciales en um.
    y_initial: float  # Posiciones iniciales en um.
    z_initial: float  # Posiciones iniciales en um.
    scantype: ScanType
    t_wait: float = 0.  # in us
    # Output
    px_size: float = None  # in um
    n_waiting_pixels: int = None
    n_pixels_total: int # (2 * self.NofPixels + 4 * self.NofAuxPixels + self.waiting_pixels)


class Piezo:
    """Piezo motor.

    Single instance for all.
    """

    _Z_ACTUATOR_RUNNING = False  # equivalent to checking Processes.ACTUATOR_Z
    _XY_ACTUATOR_RUNNING = False   # equivalent to checking Processes.ACTUATOR_XY

    def init(self):
        ...

    def _prepare_simplemove(self, x: float, y: float, z: float,
                            n_pixels_x: int = 128, n_pixels_y: int = 128, n_pixels_z: int = 128,
                            pixeltime=2000):
        """Setea los parámetros para mover a una posición x,y,z.

        El movimiento se realiza en forma suave. para cada dirección se lleva a
        cabo en n_pixel pasos, de pixeltime µs cada uno. O sea que para los
        parámetros por defecto el movimiento lleva 2000 µs * 128 pasos = 256 ms
        """
        x_f = _um2ADwin(x)
        y_f = _um2ADwin(y)
        z_f = _um2ADwin(z)

        _adw.Set_Par(_X_STEPS_PAR, n_pixels_x)
        _adw.Set_Par(_Y_STEPS_PAR, n_pixels_y)
        _adw.Set_Par(_Z_STEPS_PAR, n_pixels_z)

        _adw.Set_FPar(_X_MOVE_FPAR, x_f)
        _adw.Set_FPar(_Y_MOVE_FPAR, y_f)
        _adw.Set_FPar(_Z_MOVE_FPAR, z_f)

        _adw.Set_FPar(_TIME_PER_PIXEL_PAR, _us2A(pixeltime))

    def simple_move(self, x: float, y: float, z: float, n_pixels_x: int = 128,
                    n_pixels_y: int = 128, n_pixels_z: int = 128, pixeltime=2000):
        """Mueve suavemente a la posición pedida."""
        if self._Z_ACTUATOR_RUNNING or self._XY_ACTUATOR_RUNNING:
            _lgr.warning("No puedo moverme con el actuador andando")
            return
        self._prepare_simplemove(x, y, z, n_pixels_x, n_pixels_y, n_pixels_z, pixeltime)
        _adw.Start_Process(_Processes.MOVETO_XYZ.value)

    def get_current_position(self):
        """Return the (ADwin registered) current position."""
        return tuple(_ADwin2um((_adw.Get_FPar(par))) for par in
                     (_X_CURRENT_FPAR, _Y_CURRENT_FPAR, _Z_CURRENT_FPAR))

    def select_scan_type(self, scantype: ScanType):
        """Select scan type: xy, xz or yz."""
        fast, slow = scantype.value
        _adw.Set_FPar(_FAST_DIR_PAR, fast)
        _adw.Set_FPar(_SLOW_DIR_PAR, slow)

    def start_z_actuator(self, pixel_time: int = 1000):
        """Start Z actuator.

        Parameter
        ---------
        pixeltime: int (optional, default=1000)
            minimum time between adjustments in µs
        """
        _adw.Set_FPar(_Z_ACTUATOR_TIME, _us2A(pixel_time))
        _adw.Set_FPar(_Z_ACTUATOR_FPAR, _adw.Get_FPar(_Z_CURRENT_FPAR))
        if not self._Z_ACTUATOR_RUNNING:
            _adw.Start_Process(_Processes.ACTUATOR_Z.value)
            self._Z_ACTUATOR_RUNNING = True

    def stop_z_actuator(self):
        if self._Z_ACTUATOR_RUNNING:
            _adw.Stop_Process(_Processes.ACTUATOR_Z.value)
            self._Z_ACTUATOR_RUNNING = False
        else:
            _lgr.warning("Trying to stop inactive Z actuator")

    def start_xy_actuator(self, pixel_time: int = 1000):
        """Start XY actuator.

        Parameter
        ---------
        pixeltime: int (optional, default=1000)
            minimum time between adjustments in µs
        """
        _adw.Set_FPar(_XY_ACTUATOR_TIME, _us2A(pixel_time))
        _adw.Set_FPar(_X_ACTUATOR_FPAR, _adw.Get_FPar(_X_CURRENT_FPAR))
        _adw.Set_FPar(_Y_ACTUATOR_FPAR, _adw.Get_FPar(_Y_CURRENT_FPAR))
        if not self._XY_ACTUATOR_RUNNING:
            _adw.Start_Process(_Processes.ACTUATOR_XY.value)
            self._XY_ACTUATOR_RUNNING = True

    def stop_xy_actuator(self):
        if self._XY_ACTUATOR_RUNNING:
            _adw.Stop_Process(_Processes.ACTUATOR_XY.value)
            self._XY_ACTUATOR_RUNNING = False
        else:
            _lgr.warning("Trying to stop inactive XY actuator")

    def actuator_xy_move(self, x: float, y: float):
        """Mueve en xy usando actuador.

        No chequea que este andando el proceso, asi podemos setear las cosas
        iniciales desde antes (ver que hace el proceso)
        """
        x_f = _um2ADwin(x)
        y_f = _um2ADwin(y)
        _adw.Set_FPar(_X_ACTUATOR_FPAR, x_f)
        _adw.Set_FPar(_Y_ACTUATOR_FPAR, y_f)

    def actuator_z_move(self, z: float):
        """Mueve en z usando actuador.

        No chequea que este andando el proceso, asi podemos setear las cosas
        iniciales desde antes (ver que hace el proceso)
        """
        z_f = _um2ADwin(z)
        _adw.Set_FPar(_Z_ACTUATOR_FPAR, z_f)

    # Copiado de tools.tools
    def make_scan_data(scan_range: float, n_pixels: int, n_aux_pixels: int,
                       px_time: float, a_aux, dy: float, x_i: float, y_i: float,
                       z_i: float, scantype: str, waitingtime=0):
        """Inicializa los arreglos para realizar un escaneo suave.

        Distinguimos entre pixeles 'a medir' y 'totales', que incluyen auxiliares

        Parameters
        ----------
            scan_range (float): Largo en um de la linea.
            n_pixels (int): Numero de pixeles a medir por linea.
            n_aux_pixels (int): Numero de p'ixeles auxiliares.
            px_time (float): Tiempo por pixel en us.
            a_aux (float[4]): aceleraciones en las partes auxiliares.
            dy (float): lado de un pixel, en um.
            x_i, y_i, z_i (float): Posiciones iniciales en um.
            scantype (str): 'xy, 'xz', o 'yz'.
            waitingtime (TYPE, optional): DESCRIPTION. Defaults to 0.

        Returns
        -------
            None.
        """
        # derived parameters
        n_wt_pixels = int(waitingtime/px_time)
        px_size = scan_range/n_pixels  # en um
        v = px_size/px_time  # en um/us
        line_time = n_pixels * px_time  # en us, solo de la parte a medir
        # a_aux es la aceleracion en los pixeles auxiliares, tal vez distinto para x, y, z
        aux_time = v / a_aux
        aux_range = (1/2) * a_aux * (aux_time)**2

        dt = line_time/n_pixels  # tiempo promedio por pixel en us
        dt_aux = aux_time[0]/n_aux_pixels

        if _np.all(a_aux == _np.flipud(a_aux)) or _np.all(a_aux[0:2] == a_aux[2:4]):
            pass
        else:
            _lgr.warning('Scan signal has unmatching aux accelerations')

        # scan signal: ida, medida, frenada, vuelta, medida, frenada
        size = 4 * n_aux_pixels + 2 * n_pixels
        total_range = aux_range[0] + aux_range[1] + scan_range

        if total_range > 20:  # la platina puede recorrer hasta 20 um, pero importa la posicion inicial
            _lgr.warning('scan + aux scan excede DAC/piezo range (%s um > 20um).'
                         ' Scan signal will be saturated', total_range)
        else:
            _lgr.info('Scan signal OK')

        signal_time = _np.zeros(size)
        signal_x = _np.zeros(size)
        signal_y = _np.zeros(size)

        # smooth dy part
        signal_y[0:n_aux_pixels] = _np.linspace(0, dy, n_aux_pixels)
        signal_y[n_aux_pixels:size] = dy  # * _np.ones(size - n_aux_pixels)

        # part 1
        i0 = 0
        i1 = n_aux_pixels

        signal_time[i0:i1] = _np.linspace(0, aux_time[0], n_aux_pixels)
        t1 = signal_time[i0:i1]
        signal_x[i0:i1] = (1/2) * a_aux[0] * t1**2

        # part 2
        i2 = n_aux_pixels + n_pixels
        signal_time[i1:i2] = _np.linspace(aux_time[0] + dt,
                                          aux_time[0] + line_time, n_pixels)

        t2 = signal_time[i1:i2] - aux_time[0]
        x02 = aux_range[0]

        signal_x[i1:i2] = x02 + v * t2

        # part 3
        i3 = 2 * n_aux_pixels + n_pixels

        t3_i = aux_time[0] + line_time + dt_aux
        t3_f = aux_time[0] + aux_time[1] + line_time
        signal_time[i2:i3] = _np.linspace(t3_i, t3_f, n_aux_pixels)

        t3 = signal_time[i2:i3] - (aux_time[0] + line_time)
        x03 = aux_range[0] + scan_range

        signal_x[i2:i3] = - (1/2) * a_aux[1] * t3**2 + v * t3 + x03

        # part 4
        i4 = 3 * n_aux_pixels + n_pixels

        t4_i = aux_time[0] + aux_time[1] + line_time + dt_aux
        t4_f = aux_time[0] + aux_time[1] + aux_time[2] + line_time

        signal_time[i3:i4] = _np.linspace(t4_i, t4_f, n_aux_pixels)

        t4 = signal_time[i3:i4] - t4_i
        x04 = aux_range[0] + aux_range[1] + scan_range
        signal_x[i3:i4] = - (1/2) * a_aux[2] * t4**2 + x04

        # part 5

        i5 = 3 * n_aux_pixels + 2 * n_pixels

        t5_i = aux_time[0] + aux_time[1] + aux_time[2] + line_time + dt_aux
        t5_f = aux_time[0] + aux_time[1] + aux_time[2] + 2 * line_time

        signal_time[i4:i5] = _np.linspace(t5_i, t5_f, n_pixels)

        t5 = signal_time[i4:i5] - t5_i
        x05 = aux_range[3] + scan_range

        signal_x[i4:i5] = x05 - v * t5

        # part 6
        i6 = size

        t6_i = aux_time[0] + aux_time[1] + aux_time[2] + 2 * line_time + dt_aux
        t6_f = _np.sum(aux_time) + 2 * line_time

        signal_time[i5:i6] = _np.linspace(t6_i, t6_f, n_aux_pixels)

        t6 = signal_time[i5:i6] - t6_i
        x06 = aux_range[3]

        signal_x[i5:i6] = (1/2) * a_aux[3] * t6**2 - v * t6 + x06

        if waitingtime != 0:
            signal_x = list(signal_x)
            signal_x[i3:i3] = x04 * _np.ones(n_wt_pixels)
            signal_time[i3:i6] = signal_time[i3:i6] + waitingtime
            signal_time = list(signal_time)
            signal_time[i3:i3] = _np.linspace(t3_f, t3_f + waitingtime, n_wt_pixels)
            signal_y = _np.append(signal_y, _np.ones(n_wt_pixels) * signal_y[i3])

            signal_x = _np.array(signal_x)
            signal_time = _np.array(signal_time)

        if scantype == 'xy':
            signal_f = signal_x + x_i
            signal_s = signal_y + y_i

        if scantype == 'xz':
            signal_f = signal_x + x_i
            signal_s = signal_y + (z_i - scan_range/2)

        if scantype == 'yz':
            signal_f = signal_x + y_i
            signal_s = signal_y + (z_i - scan_range/2)

        return signal_time, signal_f, signal_s

    def prepare_scan(self, line_times, fast_positions, slow_positions):
        # TODO: implementar escaneos
        # self.waiting_pixels = int(self.waitingTime/self.pxTime)
        # self.tot_pixels = (2 * self.NofPixels + 4 * self.NofAuxPixels +
        #                    self.waiting_pixels)
        # Lo pongo como line_pixeles, espero que sea igual a x
        n_line_pixels = 80 * 3  # nro de pixeles incluyendo aceleracion
        _adw.Set_Par(_LINE_LENGTH_PAR, n_line_pixels)
        t_adwin = _us2A(line_times)
        # repeat last element because time array has to have one more
        # element than position array
        # TODO: CORREGIR EN LOS SCRIPTS ADwin
        dt = t_adwin[-1] - t_adwin[-2]
        t_adwin = _np.append(t_adwin, (t_adwin[-1] + dt,))

        x_adwin = _um2ADwin(fast_positions)
        y_adwin = _um2ADwin(slow_positions)
        # pasar todos los arrays a dtype=int32
        _adw.SetData_Long(t_adwin.ctypes.data_as(_ct.POINTER(_ct.c_int32)),
                          _SCAN_TIMES_ARRAY, 1, len(t_adwin))
        _adw.SetData_Long(x_adwin.ctypes.data_as(_ct.POINTER(_ct.c_int32)),
                          _FAST_DIR_POSITIONS_ARRAY, 1, len(x_adwin))
        _adw.SetData_Long(y_adwin.ctypes.data_as(_ct.POINTER(_ct.c_int32)),
                          _SLOW_DIR_POSITIONS_ARRAY, 1, len(y_adwin))

# ver calculate_derived_param en scan.py
    def calculate_derived_param(self, scan_range: float, n_pixels: int):
        """
        scanRange Ancho en nm
        NofPixeles es pixeles por lado
        """
        self.pxSize = scan_range/n_pixels   # in µm
        # pxTime esta en us
        # totalframeTime = n_pixels**2 * self.pxTime / 10E6  # en segundos
        # UNUSED
        # self.maxCounts = int(self.APDmaxCounts/(1/(self.pxTime*10**-6)))
        # self.linetime = (1/1000)*self.pxTime*self.NofPixels  # in ms

        #  aux scan parameters
        a_max = 4 * 10E-6  # in µm/µs^2

        # TODO: verificar esto. El codigo original es un asco
        if _np.all(self.a_aux_coeff <= 1):
            self.a_aux = self.a_aux_coeff * a_max
        else:
            self.a_aux[self.a_aux > 1] = self.a_max

        n_aux_pixels = 100
        # ver si ceil
        self.waiting_pixels = int(self.waitingTime/self.pxTime)
        self.tot_pixels = (2 * self.NofPixels + 4 * n_aux_pixels +
                           self.waiting_pixels)

        # create scan signal
        self.dy = self.pxSize

        (self.data_t, self.data_x,
         self.data_y) = self.make_scan_data(scan_range,
                                            n_pixels,
                                            n_aux_pixels,
                                            self.pxTime,
                                            self.a_aux,
                                            self.dy,
                                            self.initialPos[0],
                                            self.initialPos[1],
                                            self.initialPos[2],
                                            self.scantype,
                                            self.waitingTime)

        # Create blank image
        # full_scan = True --> size of the full scan including aux parts
        # full_scan = False --> size of the forward part of the scan
        if self.full_scan is True:
            size = (self.tot_pixels, self.tot_pixels)
        else:
            size = (self.NofPixels, self.NofPixels)
        self.image = _np.zeros(size)
        self.imageF = _np.zeros(size)
        self.imageB = _np.zeros(size)
        # self.i = 0
        # load the new parameters into the ADwin system
        self.update_device_param()

# self.x_i = self.initialPos[0]
# self.y_i = self.initialPos[1]
# self.z_i = self.initialPos[2]

# self.x_offset = 0
# self.y_offset = 0
# self.z_offset = 0

# #  load ADwin parameters

# self.adw.Set_Par(1, self.tot_pixels)

# # prepare arrays for conversion into ADwin-readable data

# self.time_range = np.size(self.data_t_adwin)
# self.space_range = np.size(self.data_x_adwin)

# self.data_t_adwin = np.array(self.data_t_adwin, dtype='int')
# self.data_x_adwin = np.array(self.data_x_adwin, dtype='int')
# self.data_y_adwin = np.array(self.data_y_adwin, dtype='int')

# self.data_t_adwin = list(self.data_t_adwin)
# self.data_x_adwin = list(self.data_x_adwin)
# self.data_y_adwin = list(self.data_y_adwin)

# self.adw.SetData_Long(self.data_t_adwin, 2, 1, self.time_range)
# self.adw.SetData_Long(self.data_x_adwin, 3, 1, self.space_range)
# self.adw.SetData_Long(self.data_y_adwin, 4, 1, self.space_range)

    def scan_line(self):
        _adw.Start_Process(_Processes.LINE_SCAN.value)
        line_time = self.data_t[-1] * 1E3  # target linetime in ms
        wait_time = line_time * 1.05 * 1E3  # ahora en segundos
        _time.sleep(wait_time)
        # while _adw.Process_Status(_Processes.LINE_SCAN.value):
        #     _time.sleep(0.05)
        line_data = self.adw.GetData_Long(1, 1, self.tot_pixels)
        return line_data

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


def init(adw: _ADwin.ADwin):
    """Initialize nanoMax piezo."""
    pos_zero = _um2ADwin(0)
    adw.Set_FPar(_X_CURRENT_FPAR, pos_zero)
    adw.Set_FPar(_Y_CURRENT_FPAR, pos_zero)
    adw.Set_FPar(_Z_CURRENT_FPAR, pos_zero)


# Initializtion
from multiprocessing import current_process
if current_process().name == 'MainProcess':
    init(_adw)  # Initialize piezo
