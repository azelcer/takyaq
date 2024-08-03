# -*- coding: utf-8 -*-
"""
Estabilizador

Separamos xy y z para estar listos
"""

import logging as _lgn
from typing import Callable as _Callable
from classes import ROI, PointInfo
from loop import StabilizerThread as _ST


_lgr = _lgn.getLogger(__name__)


class Stabilizer:
    """Wraps a stabilization procedure.

    It is just a thin forwarder wrapper around a thread, that hides direct access to
    the thread methods and private functions.
    """

    def __init__(
        self,
        camera,
        piezo,
        pixel2dist_xy: float,
        pixel2dist_z: float,
        z_ang: float,
        corrector,
        callback: _Callable[[PointInfo], None] = None,
    ):
        self._camera = camera
        self._piezo = piezo
        self._thread = _ST(camera, piezo, pixel2dist_xy,
                           pixel2dist_z, z_ang, corrector, callback)

        fromparents = dir(_ST.__base__)

        for k in dir(_ST):
            if k.startswith('_') or k in fromparents:
                continue
            setattr(self, k, getattr(self._thread, k))
    #         print(k)

    # def set_xy_rois(self, rois: list[ROI]) -> bool:
    #     return self._thread.set_xy_rois(rois)

    # def set_z_roi(self, roi: ROI) -> bool:
    #     return self._thread.set_z_roi(roi)

    # def enable_xy_tracking(self) -> bool:
    #     return self._thread.enable_xy_tracking()

    # def disable_xy_tracking(self) -> bool:
    #     return self._thread.disable_xy_tracking()

    # def set_xy_tracking(self, enabled: bool) -> bool:
    #     return self._thread.set_xy_tracking(enabled)

    # def enable_z_tracking(self) -> bool:
    #     return self._thread.enable_z_tracking()

    # def disable_z_tracking(self) -> bool:
    #     return self._thread.disable_z_tracking()

    # def set_z_tracking(self, enabled: bool) -> bool:
    #     return self._thread.set_z_tracking(enabled)

    # def enable_xy_stabilization(self) -> bool:
    #     return self._thread.enable_xy_stabilization()

    # def disable_xy_stabilization(self) -> bool:
    #     return self._thread.disable_xy_stabilization()

    # def set_xy_stabilization(self, enabled: bool) -> bool:
    #     return self._thread.set_xy_stabilization(enabled)

    # def enable_z_stabilization(self) -> bool:
    #     return self._thread.enable_z_stabilization()

    # def disable_z_stabilization(self) -> bool:
    #     return self._thread.disable_z_stabilization()

    # def set_z_stabilization(self, enabled: bool) -> bool:
    #     return self._thread.set_z_stabilization(enabled)

    # def calibrate(self, direction: str) -> bool:
    #     return self._thread.calibrate(direction)

    # def start(self) -> bool:
    #     """Inicia el loop de estabilización."""
    #     self._thread.start_loop()

    # def stop(self):
    #     self._thread.stop_loop()
    #     print("stopped")
