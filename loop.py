# -*- coding: utf-8 -*-
"""
Estabilizador

Separamos xy y z para estar listos.
Agn'ostico de GUIs'
"""

import numpy as _np
import scipy as _sp
import threading as _th
import logging as _lgn
import time as _time
from typing import Callable as _Callable
from queue import SimpleQueue as _sq
from concurrent.futures import ProcessPoolExecutor as _PPE
from typing import Union
from classes import ROI, PointInfo

from mocks import MockCamera, MockPiezo

_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


def gaussian2D(grid, amplitude, x0, y0, sigma, offset, ravel=True):
    """2D gaussian."""

    x, y = grid
    x0 = float(x0)
    y0 = float(y0)
    a = 1.0 / (2 * sigma**2)
    G = offset + amplitude * _np.exp(-(a * ((x - x0) ** 2) + a * ((y - y0) ** 2)))
    if ravel:
        G = G.ravel()
    return G


def gaussian_fit(data: _np.ndarray, x_max: float, y_max: float,
                 sigma: float) -> tuple[float, float, float]:
    """Fit a gaussian to an image.

    All data is in PIXEL units.

    Parameters
    ----------
    data : numpy.ndarray
        image as a 2D array
    x_max : float
        initial estimator of X position of the maximum
    y_max : float
        initial estimator of Y position of the maximum
    sigma : float
        initial estimator of spread of the gaussian

    Returns
    -------
    x : float
        X position of the maximum
    y : float
        y position of the maximum
    sigma : float
        FWHM

    Raises
    ------
        Should not rise
    """
    try:
        xdata = _np.meshgrid(
            _np.arange(data.shape[0]), _np.arange(data.shape[1]), indexing="ij"
        )
        v_min = data.min()
        v_max = data.max()
        # x_max, y_max = _np.unravel_index(_np.argmax(data), data.shape)
        args = (v_max - v_min, x_max, y_max, sigma, v_min)
        popt, pcov = _sp.optimize.curve_fit(gaussian2D, xdata, data.ravel(), p0=args)
    except Exception as e:
        _lgr.warning("Error fiting: %s, %s", e, type(e))
        return _np.nan, _np.nan, _np.nan
    return popt[1:4]


class StabilizerThread(_th.Thread):
    """Wraps a stabilization thread."""

    _xy_tracking: bool = False
    _z_tracking: bool = False
    _xy_stabilization: bool = False
    _z_stabilization: bool = False
    _xy_rois = None  # mins, maxs
    _z_roi = None  # min/max x, min/max y

    def __init__(
        self, out_q: _sq, camera, piezo, nmpp_xy: float, nmpp_z: float,
        callback: _Callable[[PointInfo], None] = None, *args, **kwargs
    ):
        """Init stabilization thread.

        Parameters
        ----------
            out_q:
                ...
        """
        super().__init__(*args, **kwargs)

        self._out_q = out_q
        # check if camera and piezo are OK
        if not callable(getattr(camera, "get_image", None)):
            raise ValueError("The camera object does not expose a 'get_image' method")
        self._camera = camera
        self._nmpp_xy = nmpp_xy
        self._nmpp_z = nmpp_z

        if not callable(getattr(piezo, "move", None)):
            raise ValueError("The piezo object does not expose a 'move' method")
        self._piezo = piezo
        self._stop_event = _th.Event()
        self._stop_event.set()
        self._xy_track_event = _th.Event()
        self._xy_track_event.set()
        self._z_roi_OK_event = _th.Event()
        self._z_roi_OK_event.set()
        self._cb = callback

    def set_xy_rois(self, rois: list[ROI]) -> bool:
        """Set ROIs for xy stabilization.

        Can not be used while XY tracking is active.
        """
        if self._xy_tracking:
            _lgr.warning("Trying to change xy ROIs while tracking is active")
            return False
        self._xy_rois = _np.array(
            # TODO: protect against negative numbers max(0, _.min_x), min(_.max_x, self.img.shape[1])
            [[[_.min_x, _.max_x], [_.min_y, _.max_y]] for _ in rois], dtype=_np.uint16
        )
        # TODO:  Hacer que no espere si no est'a corriendo
        return True

    def set_z_roi(self, roi: ROI) -> bool:
        """Set ROI for z stabilization.

        Can not be used while Z tracking is active.
        """
        if self._z_tracking:
            _lgr.warning("Trying to change z ROI while tracking is active")
            return False
        self._z_roi = _np.array(
            [[roi.min_x, roi.max_x], [roi.min_y, roi.max_y]], dtype=_np.uint16
        )
        self._z_roi_OK_event.clear()
        if self.is_alive():
            self._z_roi_OK_event.wait()
        return True

    def enable_xy_tracking(self) -> bool:
        """Enables tracking of XY fiduciaries."""
        if self._xy_rois is None:
            _lgr.warning("Trying to enable xy tracking without ROIs")
            return False
        if not self._z_tracking:
            self._reset_t0()
        self._initialize_last_params()
        self._xy_track_event.clear()
        self._xy_track_event.wait()
        self._xy_tracking = True
        return True

    def disable_xy_tracking(self) -> bool:
        """Disables tracking of XY fiduciaries."""
        if self._xy_stabilization:
            _lgr.warning("Trying to disable xy tracking while feedback active")
            return False
        self._xy_tracking = False
        return True

    def set_xy_tracking(self, enabled: bool) -> bool:
        if enabled:
            return self.enable_xy_tracking()
        return self.disable_xy_tracking()

    def enable_xy_stabilization(self) -> bool:
        if not self._xy_tracking:
            _lgr.warning("Trying to enable xy stabilization without tracking")
            return False
        self._xy_stabilization = True
        return True

    def disable_xy_stabilization(self) -> bool:
        if not self._xy_stabilization:
            _lgr.warning("Trying to disable xy feedback but is not active")
        self._xy_stabilization = False
        return True

    def set_xy_stabilization(self, enabled: bool) -> bool:
        if enabled:
            return self.enable_xy_stabilization()
        return self.disable_xy_stabilization()

    def enable_z_tracking(self) -> bool:
        if self._z_roi is None:
            _lgr.warning("Trying to enable z tracking without ROI")
            return False
        if not self._xy_tracking:
            self._reset_t0()
        self._z_tracking = True
        return True

    def _reset_t0(self):
        """Resetea el punto 0 del tiempo. Podría ser un parámetro externo."""
        self._t0 = _time.time()

    def disable_z_tracking(self) -> bool:
        if self._z_stabilization:
            _lgr.warning("Trying to disable z tracking while feedback active")
            return False
        self._z_tracking = False
        return True

    def set_z_tracking(self, enabled: bool) -> bool:
        if enabled:
            return self.enable_z_tracking()
        return self.disable_z_tracking()

    def enable_z_stabilization(self) -> bool:
        if not self._z_tracking:
            _lgr.warning("Trying to enable z stabilization without tracking")
            return False
        self._z_stabilization = True
        return True

    def disable_z_stabilization(self) -> bool:
        if not self._z_stabilization:
            _lgr.warning("Trying to disable z feedback but is not active")
        self._z_stabilization = False
        return True

    def set_z_stabilization(self, enabled: bool) -> bool:
        if enabled:
            return self.enable_z_stabilization()
        return self.disable_z_stabilization()

    def start_loop(self):
        """Inicia el loop de estabilización."""
        if not self._stop_event.is_set():
            _lgr.warning("Trying to start already running loop")
            return
        self._executor = _PPE()
        self._stop_event.clear()
        self.start()

    def stop_loop(self):
        """Stops the loop.

        Must be called from another thread.
        """
        self._stop_event.set()
        self.join()
        self._executor.shutdown()  ## Ver esto porque no se puede reiniciar.
        _lgr.debug("Loop ended")

    def _locate_xy_centers(self, image: _np.ndarray) -> _np.ndarray:
        """Locate centers in ROIS."""
        trimmeds = [image[roi[0, 0]:roi[0, 1], roi[1, 0]: roi[1, 1]]
                    for roi in self._xy_rois]
        x = self._last_params['x']
        y = self._last_params['y']
        s = self._last_params['s']
        rv = _np.array(tuple(self._executor.map(gaussian_fit, trimmeds, x, y, s)))
        rv = rv[:, :2] + self._xy_rois[:, 0, :]
        rv *= self._nmpp_xy
        # rv += self._xy_rois[:, 0, :] * self._nmpp_xy
        return rv

    def _initialize_last_params(self):
        # en pixeles
        # guardas (que exista imagen, etc)

        trimmeds = [self._last_image[roi[0, 0]:roi[0, 1], roi[1, 0]: roi[1, 1]]
                    for roi in self._xy_rois]
        pos_max = [_np.unravel_index(_np.argmax(data), data.shape) for data in
                   trimmeds]
        sigmas = [data.shape[0] / 3 for data in trimmeds]
        self._last_params = {
            'x': [p[0] for p in pos_max],
            'y': [p[1] for p in pos_max],
            's': sigmas,
            }

    def _locate_z_center(self, image: _np.ndarray) -> float:
        """Locate the center of the reflection used to infer Z position."""
        if self._z_roi is None:
            _lgr.error("Trying to locate z position without a ROI")
            return _np.nan
        roi = image[slice(*self._z_roi[0]), slice(*self._z_roi[1])]
        # TODO: Ver si rotar
        rv = _sp.ndimage.center_of_mass(roi)[0] * self._nmpp_z
        return rv

    def _report(self, t: float, image: _np.ndarray,
                xy_shifts: Union[_np.ndarray, None], z_shift: float):
        """Send data to provided callback."""
        if xy_shifts is None:
            xy_shifts = _np.empty((0,))
        rv = PointInfo(t, image, z_shift, xy_shifts)
        if self._cb:
            try:
                self._cb(rv)
            except Exception as e:
                _lgr.warning("Exception reporting to callback: %s(%s)", type(e), e)
        # self._out_q.put_nowait(rv)

    def run(self):
        """Main stabilization loop."""
        DELAY = .10
        initial_xy_positions = None
        initial_z_position = None
        self._t0 = _time.time()
        while not self._stop_event.is_set():
            lt = _time.monotonic()
            #  Tal vez NANs
            x_shift = 0.0
            y_shift = 0.0
            z_shift = 0.0
            xy_shifts = None
            image = self._camera.get_image()
            self._last_image = image
            t = _time.time()
            if not self._xy_track_event.is_set():
                _lgr.info("Setting xy initial positions")
                initial_xy_positions = self._locate_xy_centers(image)
                self._xy_track_event.set()
                self._xy_tracking = True                
            if not self._z_roi_OK_event.is_set():
                _lgr.info("Setting z initial positions")
                initial_z_position = self._locate_z_center(image)
                self._z_roi_OK_event.set()
                self._z_tracking = True
            if self._z_tracking:
                z_position = self._locate_z_center(image)
                z_shift = initial_z_position - z_position
            if self._xy_tracking:
                xy_positions = self._locate_xy_centers(image)
                xy_shifts = xy_positions - initial_xy_positions
            self._report(t - self._t0, image, xy_shifts, z_shift)
            if self._z_stabilization or self._xy_stabilization:
                if self._xy_stabilization:
                    x_shift, y_shift = _np.nanmean(xy_shifts, axis=0)
                    if x_shift is _np.nan:
                        _lgr.warning("x shift is NAN")
                        x_shift = 0.0
                    if y_shift is _np.nan:
                        _lgr.warning("y shift is NAN")
                        y_shift = 0.0
                if not self._z_stabilization:
                    z_shift = 0.0
                if z_shift is _np.nan:
                    _lgr.warning("z shift is NAN")
                    z_shift = 0.0
                self._piezo.move(x_shift, y_shift, z_shift)
            nt = _time.monotonic()
            delay = DELAY - (nt - lt)
            if delay < 0.001:
                delay = 0.001
            _time.sleep(delay)
        _lgr.debug("Ending loop.")


if __name__ == "__main__":
    cola = _sq()
    camera = MockCamera()
    piezo = MockPiezo()
    t = StabilizerThread(cola, camera, piezo, 25, 10)
    t.start_loop()
    _time.sleep(0.2)
    t.set_z_roi(ROI(450, 550, 450, 550))
    t.set_xy_rois([ROI(x - 30, x + 30, y - 30, y + 30) for x, y in camera.centers[:-1]])
    _time.sleep(0.2)
    t.set_z_tracking(True)
    t.set_xy_tracking(True)
    _time.sleep(0.2)
    t.set_z_stabilization(True)
    _time.sleep(1)
    t.set_xy_stabilization(True)
    _time.sleep(1)
    # print("parando")
    t.stop_loop()
    try:
        while h := cola.get(timeout=0.25):
            ...
            # print(h)
    except:
        print("Nada más en la cola")
