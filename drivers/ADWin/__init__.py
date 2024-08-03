# -*- coding: utf-8 -*-
import os as _os
from enum import Enum as _Enum

from . import ADwin


class Processes(_Enum):
    LINE_SCAN = 1  # single line scan
    MOVETO_XYZ = 2  # go to specified position and finish
    ACTUATOR_Z = 3  # continuously adjust Z
    ACTUATOR_XY = 4  # continuously adjust XY
    FLIPPER = 5
    TRACE = 6
    SHUTTERS = 7


# Conversions
def convert(x, key):
    # ADC/DAC to Voltage parameters
    m_VtoU = (0x1 << 15) / 10  #  in V^-1
    q_VtoU = 0x1 << 15
    # piezo voltage-position calibration parameters
    m_VtoL = 2.91  # in µm/V
    q_VtoL = -0.02  # in µm

#    m_VtoL = 2 # in µm/V
#    q_VtoL = 0 # in µm

    if np.any(x) < 0:
        return print('Error: x cannot take negative values')
    else:
        if key == 'VtoU': # Volts to bits
            value = x * m_VtoU + q_VtoU
            # [value] =  x * (bits/V) + bits = bits
            value = np.around(value, 0)
        if key == 'UtoV': # Bits to Volts
            value = (x - q_VtoU)/m_VtoU
            # [value] =  (x - bits)/(bits/V) = V
        if key == 'XtoU': # lenght to bits
            value = ((x - q_VtoL)/m_VtoL) * m_VtoU + q_VtoU
            # [value] =  (x - um)/(um/V) * (bits/V) + bits = bits
            value = np.around(value, 0)
        if key == 'UtoX': # bits to lenght
            value = ((x - q_VtoU)/m_VtoU) * m_VtoL + q_VtoL
            # [value] =  (x - bits)/(bits/V) * (um/V) + um = um
        if key == 'ΔXtoU': # lenght to bits
            value = (x/m_VtoL) * m_VtoU 
            # [value] =  x/(um/V) * (um/V) = bits
            value = np.around(value, 0)
        if key == 'ΔUtoX': # bits to lenght
            value = (x/m_VtoU) * m_VtoL
            # [value] =  x/(bits/V) * (um/V) = um
        if key == 'ΔVtoX': # Volts to lenght
            value = x * m_VtoL
            # [value] =  x* (um/V) = um
        if key == 'VtoX': # Volts to lenght
            value = x * m_VtoL + q_VtoL
            # [value] =  x*um/V) + um = um
        return value


def timeToADwin(t):
    "time in µs to ADwin time units of 3.33 ns"
    time_unit = 3.33E-3
    units = np.array(t/(time_unit), dtype='int')
    return units


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
_DEVICENUMBER = 0x1
_adw = ADwin.ADwin(_DEVICENUMBER, 1)
_setupDevice(_adw)
