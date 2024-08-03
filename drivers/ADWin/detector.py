# -*- coding: utf-8 -*-
from ADwin import _adw
from enum import Enum as _Enum


# Parameter that selects the detector
_DETECTOR_SELECT_PAR = 3

# Parameters that selects the fast and slow scan directions
_FAST_DIR_PAR = 10
_SLOW_DIR_PAR = 11

# piezo directions in adwin
_X_DIR = 1
_Y_DIR = 2
_Z_DIR = 6


class Detectors(_Enum):
    APD = 0
    PHOTODIODE = 1


class ScanType(_Enum):
    XY = (_X_DIR, _Y_DIR)
    XZ = (_X_DIR, _Z_DIR)
    YZ = (_Y_DIR, _Z_DIR)


def select_detector(detector: Detectors):
    _adw.Set_Par(_DETECTOR_SELECT_PAR, detector.value)


def select_scan_type(scantype: ScanType):
    fast, slow = scantype.value
    _adw.Set_FPar(_FAST_DIR_PAR, fast)
    _adw.Set_FPar(_SLOW_DIR_PAR, slow)


    # self.adw.Set_Par(1, self.tot_pixels)
    # self.data_t_adwin = tools.timeToADwin(self.data_t)
    # self.data_x_adwin = tools.convert(self.data_x, 'XtoU')
    # self.data_y_adwin = tools.convert(self.data_y, 'XtoU')

    # self.adw.SetData_Long(self.data_t_adwin, 2, 1, self.time_range)
    # self.adw.SetData_Long(self.data_x_adwin, 3, 1, self.space_range)
    # self.adw.SetData_Long(self.data_y_adwin, 4, 1, self.space_range)

#     def set_moveTo_param(self, x_f, y_f, z_f, n_pixels_x=128, n_pixels_y=128,
#                          n_pixels_z=128, pixeltime=2000):

#         x_f = tools.convert(x_f, 'XtoU')
#         y_f = tools.convert(y_f, 'XtoU')
#         z_f = tools.convert(z_f, 'XtoU')

# #        print(x_f, y_f, z_f)

#         self.adw.Set_Par(21, n_pixels_x)
#         self.adw.Set_Par(22, n_pixels_y)
#         self.adw.Set_Par(23, n_pixels_z)
#         self.adw.Set_FPar(23, x_f)
#         self.adw.Set_FPar(24, y_f)
#         self.adw.Set_FPar(25, z_f)

#         self.adw.Set_FPar(26, tools.timeToADwin(pixeltime))

# def trace_acquisition(self, Npoints, pixeltime):
#     """ 
#     Method to acquire a trace of photon counts at the current position.
#     Npoints = number of points to be acquired (max = 1024)
#     pixeltime = time per point (in μs)
#     """
#     # pixeltime in μs

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