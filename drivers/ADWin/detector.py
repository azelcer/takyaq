# -*- coding: utf-8 -*-
from . import _adw
from enum import Enum as _Enum


# Parameter that selects the detector
_DETECTOR_SELECT_PAR = 3


class Detectors(_Enum):
    APD = 0
    PHOTODIODE = 1


def select_detector(detector: Detectors):
    _adw.Set_Par(_DETECTOR_SELECT_PAR, detector.value)
