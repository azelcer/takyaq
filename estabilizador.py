# -*- coding: utf-8 -*-
"""
Estabilizador

Separamos xy y z para estar listos
"""

import numpy as _np
import scipy as _sp
import logging as _lgn
from queue import SimpleQueue as _sq
import threading as _th
from typing import Callable as _Callable
from classes import ROI, PointInfo
from loop import StabilizerThread as _ST

from mocks import MockCamera, MockPiezo, gaussian2D

_lgr = _lgn.getLogger(__name__)


# class _ReaderThread(_th.Thread):
#     def __init__(
#         self,
#         q: _sq[PointInfo | None],
#         callback: _Callable[[PointInfo], None],
#         *args,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self._q = q
#         self._cb = callback

#     def run(self):
#         while data := self._q.get():
#             if self._cb:
#                 self._cb(data)
#         print("Saliendo del loop")


class Stabilizer:
    """Wraps a stabilization procedure.

    It is just a thin wrapper around a thread.
    """

    def __init__(
        self,
        camera,
        piezo,
        pixel2dist_xy: float,
        pixel2dist_z: float,
        callback: _Callable[[PointInfo], None] = None,
    ):
        self._camera = camera
        self._piezo = piezo
        self._in_q: _sq[PointInfo | None] = _sq()
        self._thread = _ST(self._in_q, camera, piezo, pixel2dist_xy,
                           pixel2dist_z, callback)
        # self._read_th = _ReaderThread(self._in_q, callback)

    def set_xy_rois(self, rois: list[ROI]) -> bool:
        return self._thread.set_xy_rois(rois)

    def set_z_roi(self, roi: ROI) -> bool:
        return self._thread.set_z_roi(roi)

    def enable_xy_tracking(self) -> bool:
        return self._thread.enable_xy_tracking()

    def disable_xy_tracking(self) -> bool:
        return self._thread.disable_xy_tracking()

    def set_xy_tracking(self, enabled: bool) -> bool:
        return self._thread.set_xy_tracking(enabled)

    def enable_z_tracking(self) -> bool:
        return self._thread.enable_z_tracking()

    def disable_z_tracking(self) -> bool:
        return self._thread.disable_z_tracking()

    def set_z_tracking(self, enabled: bool) -> bool:
        return self._thread.set_z_tracking(enabled)

    def enable_xy_stabilization(self) -> bool:
        return self._thread.enable_xy_stabilization()

    def disable_xy_stabilization(self) -> bool:
        return self._thread.disable_xy_stabilization()

    def set_xy_stabilization(self, enabled: bool) -> bool:
        return self._thread.set_xy_stabilization(enabled)

    def enable_z_stabilization(self) -> bool:
        return self._thread.enable_z_stabilization()

    def disable_z_stabilization(self) -> bool:
        return self._thread.disable_z_stabilization()

    def set_z_stabilization(self, enabled: bool) -> bool:
        return self._thread.set_z_stabilization(enabled)

    def start(self) -> bool:
        """Inicia el loop de estabilización."""
        self._thread.start_loop()
        # self._read_th.start()

    def stop(self):
        self._thread.stop_loop()
        # self._in_q.put_nowait(None)
        # self._read_th.join()
        print("stopped")


if __name__ == "__main__":
    cola = _sq()
    camera = MockCamera()
    piezo = MockPiezo()
    t = Stabilizer(
        camera,
        piezo,
        1.0,
        1.0,
    )
    t.start()
    t.set_z_roi(ROI(450, 550, 450, 550))
    t.set_xy_rois([ROI(x - 30, x + 30, y - 30, y + 30) for x, y in camera.centers[:-1]])
    # _time.sleep(0.2)
    t.set_z_tracking(True)
    t.set_xy_tracking(True)
    # _time.sleep(0.2)
    t.set_z_stabilization(True)
    # _time.sleep(3)
    # print("parando")
    t.stop()
