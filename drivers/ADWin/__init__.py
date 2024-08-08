# -*- coding: utf-8 -*-
import os as _os
from enum import Enum as _Enum
from typing import Union as _Union
import numpy as _np
from . import ADwin


class Processes(_Enum):
    LINE_SCAN = 1  # single line scan
    MOVETO_XYZ = 2  # go to specified position and finish
    ACTUATOR_Z = 3  # continuously adjust Z
    ACTUATOR_XY = 4  # continuously adjust XY
    FLIPPER = 5
    TRACE = 6
    SHUTTERS = 7


_ADWIN_TIME_UNIT = 300


# Conversions
def microsec_to_ADwin(t: _Union[float,  _np.ndarray]):
    "time in µs to ADwin time units of 3.33 ns"
    units = _np.array(t * _ADWIN_TIME_UNIT, dtype='int')
    return units


# El DAC de Nuestra ADwin admite entradas de 16 bits, produciendo salidas entre
# -10 V y +10 V (en realidad 1 bit menos). El manual dice que valores fuera de
# rango [0, 2^16) son croppeados. 32768 debería ser 0 V.
# Existe un offset (U_Ofs)con el que se puede mover el intervalo desde [-10V; +10 V).
# Cada bit da un cambio de 20V/2^16 = 305,107 µV, por lo que el voltaje de salida
# del DAC es U_Out = U_Ofs + digital * 305.107 µV
# ADwin-Gold II.

# El piezo está calibrado en voltaje, así que podemos pasar de distancia a volt
# y de volt a valor de ADwin.
# ADC/DAC to Voltage parameters
m_VtoU = (0x1 << 15) / 10  # in V^-1
q_VtoU = 0x1 << 15  # esto corre el 0 a voltajes positivos
# piezo voltaje-posición (calibrado por alguien)
um_per_volt = 2.91  # proporcionalidad en  µm/V
um_shift = -0.02  # shift fijo en µm. Es muy rato que sea igual para todos


# TODO: protegerse contra entradas negativas.

def um2ADwin(x: _Union[float, _np.ndarray]) -> int:
    """x en µm"""
    # TODO: ver si redondear
    rv = _np.array(((x - um_shift) / um_per_volt) * m_VtoU + q_VtoU, dtype='int')
    return rv


def ADwin2um(x:_Union[int, _np.ndarray]) -> float:
    """rv en µm"""
    rv = _np.array(um_per_volt * (x - q_VtoU) / m_VtoU + um_shift)
    return rv


def convert(x, key):
    # ADC/DAC to Voltage parameters
    m_VtoU = (0x1 << 15) / 10  # in V^-1, sólo usamos media escala
    q_VtoU = 0x1 << 15
    # piezo voltage-position calibration parameters
    m_VtoL = 2.91  # in µm/V
    q_VtoL = -0.02  # in µm
    if _np.any(x) < 0:
        return print('Error: x cannot take negative values')
    else:
        if key == 'VtoU':  # Volts to bits
            value = x * m_VtoU + q_VtoU
            # [value] =  x * (bits/V) + bits = bits
            value = _np.around(value, 0)
        if key == 'UtoV':  # Bits to Volts
            value = (x - q_VtoU)/m_VtoU
            # [value] =  (x - bits)/(bits/V) = V
        if key == 'ΔXtoU':  # lenght to bits
            value = (x/m_VtoL) * m_VtoU
            # [value] =  x/(um/V) * (um/V) = bits
            value = _np.around(value, 0)
        if key == 'ΔUtoX':  # bits to lenght
            value = (x/m_VtoU) * m_VtoL
            # [value] =  x/(bits/V) * (um/V) = um
        if key == 'ΔVtoX':  # Volts to lenght
            value = x * m_VtoL
            # [value] =  x* (um/V) = um
        if key == 'VtoX':  # Volts to lenght
            value = x * m_VtoL + q_VtoL
            # [value] =  x*um/V) + um = um
        return value


def _setupDevice(adw: ADwin.ADwin):
    """Load programs into adwin."""

    BTL = "ADwin11.btl"
    PROCESS_1 = "line_scan.TB1"
    PROCESS_2 = "moveto_xyz.TB2"
    PROCESS_3 = "actuator_z.TB3"
    PROCESS_4 = "actuator_xy.TB4"
    PROCESS_5 = "flipper.TB5"
    PROCESS_6 = "trace.TB6"
    PROCESS_7 = "shutters.TB7"

    adw.Boot(_os.path.join(adw.ADwindir, BTL))

    currdir = _os.getcwd()
    process_folder = _os.path.join(currdir, "processes")

    process_1 = _os.path.join(process_folder, PROCESS_1)
    process_2 = _os.path.join(process_folder, PROCESS_2)
    process_3 = _os.path.join(process_folder, PROCESS_3)
    process_4 = _os.path.join(process_folder, PROCESS_4)
    process_5 = _os.path.join(process_folder, PROCESS_5)
    process_6 = _os.path.join(process_folder, PROCESS_6)
    process_7 = _os.path.join(process_folder, PROCESS_7)

    adw.Load_Process(process_1)
    adw.Load_Process(process_2)
    adw.Load_Process(process_3)
    adw.Load_Process(process_4)
    adw.Load_Process(process_5)
    adw.Load_Process(process_6)
    adw.Load_Process(process_7)


# Initializtion
from multiprocessing import current_process
_DEVICENUMBER = 0x1
_adw = ADwin.ADwin(_DEVICENUMBER, 1)  # TODO: Exportar este símbolo
if current_process().name == 'MainProcess':
    _setupDevice(_adw)
