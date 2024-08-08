# -*- coding: utf-8 -*-
"""
Estabilizador

Separamos xy y z para estar listos.
Agnóstico de GUIs'
"""

import numpy as _np
import scipy as _sp
import threading as _th
import logging as _lgn
import time as _time
import os as _os
from typing import Callable as _Callable
from concurrent.futures import ProcessPoolExecutor as _PPE
from typing import Union
from classes import ROI, PointInfo


_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


def _gaussian2D(grid, amplitude, x0, y0, sigma, offset, ravel=True):
    """Generate a 2D gaussian.

    Parameters
    ----------
    grid: numpy.ndarray
        X, Y coordinates grid to generate the gaussian over
    amplitude: float
        FIXME: amplitude
    x0, y0: float
        position of gaussian center
    sigma: float
        FWHM
    offset:
        uniform background value
    ravel: bool, default True
        If True, returns the raveled values, otherwise return a 2D array
    """

    x, y = grid
    x0 = float(x0)
    y0 = float(y0)
    a = 1.0 / (2 * sigma**2)
    G = offset + amplitude * _np.exp(-(a * ((x - x0) ** 2) + a * ((y - y0) ** 2)))
    if ravel:
        G = G.ravel()
    return G


def _gaussian_fit(data: _np.ndarray, x_max: float, y_max: float,
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

    If case of an error, returns numpy.nan for every value

    Raises
    ------
        Should not raise
    """
    try:
        xdata = _np.meshgrid(
            _np.arange(data.shape[0]), _np.arange(data.shape[1]), indexing="ij"
        )
        v_min = data.min()
        v_max = data.max()
        args = (v_max - v_min, x_max, y_max, sigma, v_min)
        popt, pcov = _sp.optimize.curve_fit(_gaussian2D, xdata, data.ravel(), p0=args)
    except Exception as e:
        _lgr.warning("Error fiting: %s, %s", e, type(e))
        return _np.nan, _np.nan, _np.nan
    return popt[1:4]


class StabilizerThread(_th.Thread):
    """Wraps a stabilization thread."""

    # Status flags
    _xy_tracking: bool = False
    _z_tracking: bool = False
    _xy_stabilization: bool = False
    _z_stabilization: bool = False
    # ROIS from the user
    _xy_rois: _np.ndarray = None  # [ [min, max]_x, [min, max]_y] * n_rois
    _z_roi = None  # min/max x, min/max y
    _last_image: _np.ndarray = _np.empty((50, 50))
    _pos = _np.zeros((3,))  # current position in nm
    _period = 0.150  # minumum loop time in seconds

    def __init__(
        self, camera, piezo, nmpp_xy: float, nmpp_z: float, z_ang: float,
        corrector, callback: _Callable[[PointInfo], None] = None, *args, **kwargs
    ):
        """Init stabilization thread.

        Parameters
        ----------
        camera:
            Camera. Must implement a method called `get_image`, that returns
            a 2d numpy.ndarray representing the image
        piezo:
            Piezo controller. Must implement a method called `set_position` that
            accepts x, y and z positions
        nmpp_xy: float
            nanometers in XY plane per camera pixel
        nmpp_z: float
            nanometers in Z direction per camera pixel
        z_ang: float
            angle between positive X direction and movement of the Z spot, in radians
        corrector:
            object that provides a response
        callback: Callable
            Callable to report measured shifts. Will receive a `PointInfo`
            object as the only parameter
        """
        super().__init__(*args, **kwargs)

        # check if camera and piezo are OK
        if not callable(getattr(camera, "get_image", None)):
            raise ValueError("The camera object does not expose a 'get_image' method")
        self._camera = camera
        self._nmpp_xy = nmpp_xy
        self._nmpp_z = nmpp_z
        self._rot_vec = _np.array((_np.cos(z_ang), _np.sin(z_ang), ))

        if not callable(getattr(piezo, "set_position", None)):
            raise ValueError("The piezo object does not expose a 'set_position' method")
        if not callable(getattr(piezo, "get_position", None)):
            raise ValueError("The piezo object does not expose a 'get_position' method")
        self._piezo = piezo
        self._stop_event = _th.Event()
        self._stop_event.set()
        # FIXME: ROI setting and tracking are coupled, and names are mixed up
        self._xy_track_event = _th.Event()
        self._xy_track_event.set()
        self._z_roi_OK_event = _th.Event()
        self._z_roi_OK_event.set()
        self._calibrate_event = _th.Event()
        self._rsp = corrector
        self._cb = callback

    def set_log_level(self, loglevel: int):
        if loglevel < 0:
            _lgr.warning("Invalid log level asked: %s", loglevel)
        else:
            _lgr.setLevel(loglevel)

    def set_min_period(self, period: float):
        """Set minimum period between position adjustments.

        Parameters
        ----------
        period: float
            Minimum period, in seconds.

        The period is not precise, and might be longer than asked for if the
        time needed to locate the stage real position takes too long.

        The thread always sleeps for at least 10 ms, in order to let other
        threads run.

        Raises
        ------
        ValuError if requested period is negative
        """
        if period < 0:
            raise ValueError(f"Period can not be negative ({period})")
        self._period = period

    def set_xy_rois(self, rois: list[ROI]) -> bool:
        """Set ROIs for xy stabilization.

        Can not be used while XY tracking is active.

        Parameters
        ----------
        rois: list[ROI]
            list of XY rois

        Return
        ------
        True if successful, False otherwise
        """
        if self._xy_tracking:
            _lgr.warning("Trying to change xy ROIs while tracking is active")
            return False
        self._xy_rois = _np.array(
            # TODO: protect against negative numbers max(0, _.min_x), min(_.max_x, self.img.shape[1])
            [[[_.min_x, _.max_x], [_.min_y, _.max_y]] for _ in rois], dtype=_np.uint16
        )
        return True

    def set_z_roi(self, roi: ROI) -> bool:
        """Set ROI for z stabilization.

        Can not be used while Z tracking is active.

        Parameter
        ---------
        roi: ROI
            Z roi

        Return
        ------
        True if successful, False otherwise
        """
        if self._z_tracking:
            _lgr.warning("Trying to change z ROI while tracking is active")
            return False
        # TODO: protect against negative numbers max(0, _.min_x), min(_.max_x, self.img.shape[1])
        self._z_roi = _np.array(
            [[roi.min_x, roi.max_x], [roi.min_y, roi.max_y]], dtype=_np.uint16
        )
        self._z_roi_OK_event.clear()
        if self.is_alive():
            self._z_roi_OK_event.wait()
        return True

    def enable_xy_tracking(self) -> bool:
        """Enable tracking of XY fiduciaries."""
        if self._xy_rois is None:
            _lgr.warning("Trying to enable xy tracking without ROIs")
            return False

        self._initialize_last_params()
        self._xy_track_event.clear()
        self._xy_track_event.wait()
        self._xy_tracking = True
        return True

    def disable_xy_tracking(self) -> bool:
        """Disable tracking of XY fiduciaries."""
        if self._xy_stabilization:
            _lgr.warning("Trying to disable xy tracking while feedback active")
            return False
        self._xy_tracking = False
        return True

    def set_xy_tracking(self, enabled: bool) -> bool:
        """Set XY tracking ON or OFF."""
        if enabled:
            return self.enable_xy_tracking()
        return self.disable_xy_tracking()

    def enable_xy_stabilization(self) -> bool:
        """Enable stabilization on XY plane."""
        if not self._xy_tracking:
            _lgr.warning("Trying to enable xy stabilization without tracking")
            return False
        self._rsp.reset_xy(len(self._xy_rois))
        self._xy_stabilization = True
        return True

    def disable_xy_stabilization(self) -> bool:
        """Disable stabilization on XY plane."""
        if not self._xy_stabilization:
            _lgr.warning("Trying to disable xy feedback but is not active")
        self._xy_stabilization = False
        return True

    def set_xy_stabilization(self, enabled: bool) -> bool:
        """Set XY stabilization ON or OFF."""
        if enabled:
            return self.enable_xy_stabilization()
        return self.disable_xy_stabilization()

    def enable_z_tracking(self) -> bool:
        """Enable tracking of Z position."""
        if self._z_roi is None:
            _lgr.warning("Trying to enable z tracking without ROI")
            return False

        self._z_tracking = True
        return True

    def disable_z_tracking(self) -> bool:
        """Disable tracking of Z position."""
        if self._z_stabilization:
            _lgr.warning("Trying to disable z tracking while feedback active")
            return False
        self._z_tracking = False
        return True

    def set_z_tracking(self, enabled: bool) -> bool:
        """Set tracking of Z position ON or OFF."""
        if enabled:
            return self.enable_z_tracking()
        return self.disable_z_tracking()

    def enable_z_stabilization(self) -> bool:
        """Enable stabilization of Z position."""
        if not self._z_tracking:
            _lgr.warning("Trying to enable z stabilization without tracking")
            return False
        self._rsp.reset_z()
        self._z_stabilization = True
        return True

    def disable_z_stabilization(self) -> bool:
        """Disable stabilization of Z position."""
        if not self._z_stabilization:
            _lgr.warning("Trying to disable z feedback but is not active")
        self._z_stabilization = False
        return True

    def calibrate(self, direction: str) -> bool:
        """Perform calibration of pixel size."""
        if direction not in ['x', 'y', 'z']:
            _lgr.warning("Invalid calibration direction: %s", direction)
            return False

        # no `match` yet (we support python 3.9), so do a dirty trick
        self._calib_idx = {'x': 0, 'y': 1, 'z': 2}[direction]
        self._calibrate_event.set()

    def set_z_stabilization(self, enabled: bool) -> bool:
        """Set stabilization of Z position ON or OFF."""
        if enabled:
            return self.enable_z_stabilization()
        return self.disable_z_stabilization()

    def start_loop(self) -> bool:
        """Start tracking and stabilization loop."""
        if not self._stop_event.is_set():
            _lgr.warning("Trying to start already running loop")
            return False
        self._executor = _PPE()
        # prime pool for responsiveness (a _must_ on windows).
        nproc = _os.cpu_count()
        params = [[_np.eye(3)] * nproc, [1.] * nproc, [1.] * nproc, [1.] * nproc]
        _ = tuple(self._executor.map(_gaussian_fit, *params))
        self._stop_event.clear()
        self.start()
        return True

    def stop_loop(self):
        """Stop tracking and stabilization loop and release resources.

        Must be called from another thread to avoid deadlocks.
        """
        self._stop_event.set()
        self.join()
        self._executor.shutdown()
        _lgr.debug("Loop ended")

    def _locate_xy_centers(self, image: _np.ndarray) -> _np.ndarray:
        """Locate centers in XY ROIS.

        Returns values in pixels

        Parameter
        ---------
        image: numpy.ndarray
            2D array with the image to process

        Return
        ------
            numpy.ndarray of shape (NROIS, 2) with x,y center in nm
        """
        trimmeds = [image[roi[0, 0]:roi[0, 1], roi[1, 0]: roi[1, 1]]
                    for roi in self._xy_rois]
        x = self._last_params['x']
        y = self._last_params['y']
        s = self._last_params['s']
        locs = _np.array(tuple(self._executor.map(_gaussian_fit, trimmeds, x, y, s)))
        self._last_params['x'] = locs[:, 0]
        nanloc = _np.isnan(locs[:, 0])  # if x is nan, y also is nan
        self._last_params['x'][nanloc] = x[nanloc]
        self._last_params['y'] = locs[:, 1]
        self._last_params['y'][nanloc] = y[nanloc]
        self._last_params['s'] = locs[:, 2]
        self._last_params['s'][nanloc] = s[nanloc]
        rv = locs[:, :2] + self._xy_rois[:, 0, :]
        rv *= self._nmpp_xy
        return rv

    def _initialize_last_params(self):
        """Initialize fitting parameters.

        All values are *in pixels* inside each ROI.

        TODO: Protect against errors (image must exist, ROIS must fit into image, etc.)
        """
        trimmeds = [self._last_image[roi[0, 0]:roi[0, 1], roi[1, 0]: roi[1, 1]]
                    for roi in self._xy_rois]
        pos_max = [_np.unravel_index(_np.argmax(data), data.shape) for data in
                   trimmeds]
        sigmas = [data.shape[0] / 3 for data in trimmeds]
        self._last_params = {
            'x': _np.array([p[0] for p in pos_max], dtype=float),
            'y': _np.array([p[1] for p in pos_max], dtype=float),
            's': _np.array(sigmas, dtype=float),
            }

    def _locate_z_center(self, image: _np.ndarray) -> float:
        """Locate the center of the reflection used to infer Z position."""
        if self._z_roi is None:
            _lgr.error("Trying to locate z position without a ROI")
            return _np.nan
        roi = image[slice(*self._z_roi[0]), slice(*self._z_roi[1])]
        # ang is measured counterclockwise from the X axis. We rotate *clockwise*
        rv = _np.sum(_np.array(_sp.ndimage.center_of_mass(roi)) * self._rot_vec) * self._nmpp_z
        return rv

    def _move_relative(self, dx: float, dy: float, dz: float):
        """Perform a relative movement."""
        self._pos[0] += dx
        self._pos[1] += dy
        self._pos[2] += dz
        self._piezo.set_position(*self._pos)

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

    def _calibrate_xy(self, length: float, initial_xy_positions: _np.ndarray,
                      points: int = 20):
        """Calibrate nm per pixel in XY plane.

        Runs it own small loop.
        moves around current position
        TODO: Inform about XY coupling (camera rotation?)
        WARNING: Z is handled separately

        Parameters
        ----------
        length: float
            calibration displacement in nm
        initial_xy_positions: numpy.ndarray
            initial positions of the fiduciary marks
        points: int, default 20
            number of calibration points
        """
        c_idx = self._calib_idx  # 0 for X, 1 for Y, Z is complicated
        shifts, step = _np.linspace(-length/2., length/2., points, retstep=True)
        response = _np.empty_like(shifts)
        if self._xy_rois is None:
            _lgr.warning("Trying to calibrate xy without ROIs")
            return False
        if not self._xy_track_event.is_set():
            _lgr.warning("Trying to calibrate xy without tracking")
            return False
        oldpos = _np.copy(self._pos)
        rel_vec = _np.zeros((3,))
        try:
            rel_vec[c_idx] = -length/2.
            self._move_relative(*rel_vec)
            rel_vec[c_idx] = step
            image = self._camera.get_image()
            self._initialize_last_params()  # we made a LARGE shift
            for idx, s in enumerate(shifts):
                image = self._camera.get_image()
                xy_shifts = self._locate_xy_centers(image)
                self._report(_time.time(), image, xy_shifts - initial_xy_positions, 0)
                x = _np.nanmean(xy_shifts[:, c_idx])
                response[idx] = x / self._nmpp_xy
                self._move_relative(*rel_vec)
                _time.sleep(.050)
            # TODO: better reporting
            for x, y in zip(shifts, response):
                print(f"{x}, {y}")
            vec, _ = _np.linalg.lstsq(_np.vstack([shifts, _np.ones(points)]).T,
                                      response, rcond=None)[0]
            print("slope = ", 1/vec)
        except Exception as e:
            _lgr.warning("Exception calibrating x: %s(%s)", type(e), e)
        self._pos[:] = oldpos
        self._piezo.set_position(*self._pos)

    def _calibrate_z(self, length: float, initial_xy_positions: _np.ndarray,
                     points: int = 20):
        """Calibrate Znm per pixel.

        Runs its own loop.

        Parameters
        ----------
        length: float
            calibration displacement in nm
        initial_xy_positions: numpy.ndarray
            initial positions of the fiduciary marks
        points: int, default 20
            number of calibration points
        """
        shifts, step = _np.linspace(-length/2., length/2., points, retstep=True)
        response = _np.empty((points, 2, ))
        if self._z_roi is None:
            _lgr.warning("Trying to calibrate z without ROI")
            return False
        if not self._z_roi_OK_event.is_set():
            _lgr.warning("Trying to calibrate z without tracking")
            return False
        oldpos = _np.copy(self._pos)
        rel_vec = _np.zeros((3,))
        try:
            rel_vec[2] = -length/2.
            self._move_relative(*rel_vec)
            rel_vec[2] = step
            image = self._camera.get_image()
            for idx, s in enumerate(shifts):
                image = self._camera.get_image()
                roi = image[slice(*self._z_roi[0]), slice(*self._z_roi[1])]
                c = _np.array(_sp.ndimage.center_of_mass(roi))
                xy_data = (None if self._xy_rois is None else
                           self._locate_xy_centers(image) - initial_xy_positions)
                self._report(_time.time(), image, xy_data, _np.nan)
                response[idx] = c
                self._move_relative(*rel_vec)
                _time.sleep(.050)
            # TODO: better reporting
            for x, y in zip(shifts, response):
                print(f"{x}, {y}")
            vec, _ = _np.linalg.lstsq(_np.vstack([shifts, _np.ones(points)]).T,
                                        response, rcond=None)[0]
            print("slope = ", 1/vec)
            print("nmpp z = ", _np.sum(1./vec**2)**.5)
            # watch out order
            print("Angle(rad) = ", _np.arctan2(vec[1], vec[0]))
        except Exception as e:
            _lgr.warning("Exception calibrating z: %s(%s)", type(e), e)
        self._pos[:] = oldpos
        self._piezo.set_position(*self._pos)

    def run(self):
        """Run main stabilization loop."""
        # TODO: let delay be configurable
        DELAY = self._period
        initial_xy_positions = None
        initial_z_position = None
        while not self._stop_event.is_set():
            lt = _time.monotonic()
            z_shift = 0.0
            xy_shifts = None
            if self._calibrate_event.is_set():
                _lgr.debug("Calibration event received")
                if self._calib_idx >= 0 and self._calib_idx < 2:
                    self._calibrate_xy(50., initial_xy_positions)
                elif self._calib_idx ==2:
                    self._calibrate_z(20., initial_xy_positions)
                else:
                    _lgr.warning("Invalid calibration direction detected")
                self._calibrate_event.clear()
            try:
                image = self._camera.get_image()
                t = _time.time()
                self._last_image = image
            except Exception as e:
                _lgr.error("Could not acquire image: %s (%s)", type(e), e)
                image = _np.diag(_np.full(max(*self._last_image.shape), 255))
                t = _time.time()
                self._report(t, image,
                             _np.full_like(initial_xy_positions, _np.nan), _np.nan)
                _time.sleep(DELAY)
                continue
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
            self._report(t, image, xy_shifts, z_shift)
            if self._z_stabilization or self._xy_stabilization:
                if z_shift is _np.nan:
                    _lgr.warning("z shift is NAN")
                    z_shift = 0.0
                try:
                    x_resp, y_resp, z_resp = self._rsp.response(t, xy_shifts, z_shift)
                except Exception as e:
                    _lgr.warning("Error getting correction: %s, %s", e, type(e))
                    x_resp = y_resp = z_resp = 0.0
                if not self._z_stabilization:
                    z_resp = 0.0
                if not self._xy_stabilization:
                    x_resp = y_resp = 0.0
                self._move_relative(x_resp, y_resp, z_resp)
            nt = _time.monotonic()
            delay = DELAY - (nt - lt)
            if delay < 0.001:  # be nice to other threads
                delay = 0.001
            _time.sleep(delay)
        _lgr.debug("Ending loop.")
