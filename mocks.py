#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mocks
"""
import numpy as _np
import time as _time
# from numba import njit as _njit


# @_njit
def gaussian2D(grid, amplitude, x0, y0, sigma, offset):
    """2D gaussian."""

    x, y = grid
    x0 = float(x0)
    y0 = float(y0)
    a = 1.0 / (2 * sigma**2)
    G = offset + amplitude * _np.exp(-(a * ((x - x0) ** 2) + a * ((y - y0) ** 2)))
    return G


class MockCamera:
    """Mock camera for testing and development.

    Simulates a drift that:
        - Shifts X position in a sinusoidal way with period 1.
        - Shifts Y position in a sinusoidal way with period e.
        - Shifts Z position in a sinusoidal way with period pi.

    Moreover, it adds random noise to the position of the fiducial markers and a
    background noise.
    """

    max_x = 1200
    max_y = 900
    centers = (
        (100, 100),
        (100, 300),
        (100, 500),
        (100, 700),
        (300, 100),
        (300, 300),
        (300, 500),
        (300, 700),
        (500, 100),
        (500, 300),
        (500, 500),
        (500, 700),
        (800, 800),
    )
    sigma = 200.  # FWHM of the signals, in nm
    _X_PERIOD = 4
    _Y_PERIOD = _np.e*4
    _Z_PERIOD = _np.pi*4
    _shifts = _np.zeros((3,), dtype=_np.float64)

    grid = _np.array(_np.meshgrid(_np.arange(max_x), _np.arange(max_y), indexing="ij"))
    f = True
    def __init__(self, nmpp_x, nmpp_y, nmpp_z, sigma, noise_level=3, drift_amplitude=3):
        """Init Mock camera.

        TODO: Improve Z movement mocking to include the physical subtleties (angle). So far,
        it only shifts in the Z direction

        Parameters
        ----------
        nmpp_x : float
            nanometers per pixel in X direction.
        nmpp_y : float
            nanometers per pixel in X direction.
        nmpp_z : float
            nanometers per pixel in Z direction (see comment).
        sigma : float
            FWHM of the signals, in nm.
        noise_level : float, optional
            Random shifts of XYZ positions in nm. The default is 3.
        drift_amplitude : float, optional
            Amplitude of the periodic shift in nm. The default is 3.
        """
        self._nl = noise_level
        self._drift = drift_amplitude
        self._nmpp_x = nmpp_x
        self._nmpp_y = nmpp_y
        self._nmpp_z = nmpp_z
        self.sigma = sigma

    def get_image(self):
        """Return a faked image."""
        # Some random noise
        # if not self.f:
        #     if _np.random.random_sample() > 0.9:  # falla una de cada 10
        #         raise ValueError("error en camara")
        # self.f = False
        rv = _np.random.poisson(
            2.5,
            (
                self.max_x,
                self.max_y,
            ),
        ).astype(_np.float64)
        t = _time.monotonic()
        # limit gaussian creation to +-4 sigma from center for speed
        slice_size = int(self.sigma / self._nmpp_x * 4)  # in pixels
        for x0, y0 in self.centers[:-1]:
            x0 += self._shifts[0] / self._nmpp_x
            y0 += self._shifts[1] / self._nmpp_y
            x0 += (
                (_np.random.random_sample() - 0.5) * self._nl
                + _np.sin(t / self._X_PERIOD * 2 * _np.pi) * self._drift
            ) / self._nmpp_x
            y0 += (
                (_np.random.random_sample() - 0.5) * self._nl
                + _np.sin(t / self._Y_PERIOD * 2 * _np.pi) * self._drift
            ) / self._nmpp_y
            cx = int(x0)
            cy = int(y0)
            slicex = slice(max(cx - slice_size, 0), min(cx + slice_size, self.max_x))
            slicey = slice(max(cy - slice_size, 0), min(cy + slice_size, self.max_y))
            rv[slicex, slicey] += gaussian2D(
                self.grid[:, slicex, slicey], 550, x0, y0, self.sigma / self._nmpp_x, 0
            )
        x0, y0 = self.centers[-1]
        x0 += (
            (_np.random.random_sample() - 0.5) * self._nl
            + (
                2 * abs(2 * (t / self._Z_PERIOD - _np.floor(t / self._Z_PERIOD + 0.5)))
                - 1
            )
            * self._drift
        ) / self._nmpp_z
        x0 += self._shifts[2] / self._nmpp_z
        cx = int(x0)
        cy = int(y0)
        slicex = slice(max(cx - slice_size, 0), min(cx + slice_size, self.max_x))
        slicey = slice(max(cy - slice_size, 0), min(cy + slice_size, self.max_y))
        rv[slicex, slicey] += gaussian2D(  # use X coordinate nmpp, since it maps OK
            self.grid[:, slicex, slicey], 550, x0, y0, self.sigma / self._nmpp_x, 0
        )
        return rv.astype(_np.uint16)

    def shift(self, dx: float, dy: float, dz: float):
        """Shift origin of coordinates.

        Simulates a stage movement.
        """
        self._shifts += _np.array(
            (
                dx,
                dy,
                dz,
            )
        )


class MockPiezo:
    """Mock piezoelectric motor.

    It can shift the zero of the mock camera, to test stabilization strategies.
    """

    lastime = _time.monotonic()

    def __init__(self, camera=None):
        self._camera = camera

    def move(self, *args):
        t = _time.monotonic()
        # print("loop took:", t-self.lastime, "shifts=", args)
        if self._camera:
            self._camera.shift(
                args[0], args[1], args[2]
            )
        self.lastime = t
        return
