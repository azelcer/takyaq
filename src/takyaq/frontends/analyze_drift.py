# -*- coding: utf-8 -*-
"""

"""
import numpy as _np
import scipy as _sp
import time as _time
from typing import Optional as _Optional, BinaryIO as _BinaryIO

import warnings
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, Qt
from PyQt5.QtWidgets import (
    QGroupBox,
    QFrame,
    QApplication,
    QGridLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QHBoxLayout,
    QVBoxLayout,
    QLineEdit,
    QDoubleSpinBox,
)

import pyqtgraph as _pg

import logging as _lgn

_lgr = _lgn.getLogger(__name__)


def slicex(x, n):
    if n == 0:
        return x
    if n > 0:
        return x[n:]
    if n < 0:
        return x[:n]


def nan_correlate(a, b):
    if (not (len(a) == len(b))):
        raise IOError("input must have same length")
    L = len(a)
    offsets = _np.arange(-L+1, L)
    corrs = _np.array([_np.nansum(slicex(a, l) * slicex(b, -l)) for l in offsets])
    return offsets, corrs


class Frontend(QFrame):
    """PyQt Frontend for Takyaq.

    Implemented as a QFrame so it can be easily integrated within a larger app.
    """

    _x_plots = []
    _y_plots = []
    # _z_plot

    def __init__(self, *args, **kwargs):
        """Init Frontend."""

        super().__init__(*args, **kwargs)
        self.setup_gui()
        # Callback object
        try:
            self._load_data()
        except Exception as e:
            _lgr.error("NO pude abrir el archivo")
            print(e, type(e))
            pass

    def reset_graphs(self, roi_len: int):
        """Reset graphs contents and adjust to number of XY ROIs."""
        neplots = len(self._x_plots)
        if roi_len > neplots:
            self._x_plots.extend(
                [
                    self.xyzGraph.xPlot.plot(
                        pen="w", alpha=0.3, auto=False, connect="finite"
                    )
                    for _ in range(roi_len - neplots)
                ]
            )
            self._y_plots.extend(
                [
                    self.xyzGraph.yPlot.plot(
                        pen="w", alpha=0.3, auto=False, connect="finite"
                    )
                    for _ in range(roi_len - neplots)
                ]
            )
        elif roi_len < neplots:
            for i in range(neplots - roi_len):
                self.xyzGraph.xPlot.removeItem(self._x_plots.pop())
                self.xyzGraph.yPlot.removeItem(self._y_plots.pop())
        for p in self._x_plots:
            p.clear()
            p.setAlpha(0.3, auto=False)
        for p in self._y_plots:
            p.clear()
            p.setAlpha(0.3, auto=False)
        self.xmeanCurve.setZValue(roi_len)
        self.ymeanCurve.setZValue(roi_len)

    def _load_data(self):
        with open("/tmp/xy_data.npy", 'rb') as fd:
            data = []
            n_batches = 0
            try:
                while True:
                    data.append(_np.load(fd))
                    n_batches += 1
            except EOFError:
                print(f"Loaded {n_batches} of lenghts {[len(x) for x in data]}")
            self._data = _np.concatenate(data) if data else None
            if self._data is not None:
                self._data['t'] -= self._data['t'][0]
        with open("/tmp/z_data.npy", 'rb') as fd:
            data = []
            n_batches = 0
            try:
                while True:
                    data.append(_np.load(fd))
                    n_batches += 1
            except EOFError:
                print(f"Loaded {n_batches} of lenghts {[len(x) for x in data]}")
            self._z_data = _np.concatenate(data) if data else None
            if self._z_data is not None:
                self._z_data['t'] -= self._z_data['t'][0]
        if self._data is not None:
            print("cargamos", len(self._data), " puntos xy.")
            print("El numero de ROIs es de", len(self._data[0]['xy']), " puntos.")
            self.reset_graphs(len(self._data[0]['xy']))
        self.plot_data()

    def plot_data(self):
        if self._z_data is not None:
            z_data = self._z_data['z']
            z_t_data = self._z_data['t']# - self._z_data['t'][0]
            self.zstd_value.setText(f"{_np.nanstd(z_data):.2f}")
            # update Graphs
            self.zCurve.setData(z_t_data, z_data)
            try:  # It is possible to have all NANs data
                hist, bin_edges = _np.histogram(z_data, bins=30,
                                                range=(_np.nanmin(z_data),
                                                       _np.nanmax(z_data)))
                self.zHistogram.setOpts(
                    x=_np.mean((bin_edges[:-1], bin_edges[1:],), axis=0),
                    height=hist,
                    width=bin_edges[1]-bin_edges[0]
                    )
            except Exception as e:
                print("Excepcion ploteando z:", e, (type(e)))

        if self._data is not None:
            x_data = self._data['xy'][:, :, 0]
            y_data = self._data['xy'][:, :, 1]
            t_data = self._data['t']# - self._data['t'][0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_mean = _np.nanmean(x_data, axis=1)
                y_mean = _np.nanmean(y_data, axis=1)
                # update reports
                self.xstd_value.setText(
                    f"{_np.nanstd(x_mean):.2f} - {_np.nanmean(_np.nanstd(x_data, axis=0)):.2f}"
                )
                self.ystd_value.setText(
                    f"{_np.nanstd(y_mean):.2f} - {_np.nanmean(_np.nanstd(y_data, axis=0)):.2f}"
                )
            # # update Graphs
            for i, p in enumerate(self._x_plots):
                p.setData(t_data, x_data[:, i])
                # print(t_data)
            self.xmeanCurve.setData(t_data, x_mean)
            for i, p in enumerate(self._y_plots):
                p.setData(t_data, y_data[:, i])
            self.ymeanCurve.setData(t_data, y_mean)
            self.xyDataItem.setData(x_mean, y_mean)
            x_off, x_corr = nan_correlate(x_mean, x_mean)
            y_off, y_corr = nan_correlate(y_mean, y_mean)
            dt = t_data[-1] - t_data[-2]
            self.xacplot.setData(x_off* dt, x_corr)
            self.yacplot.setData(y_off * dt, y_corr)

    def setup_gui(self):
        """Create and lay out all GUI objects."""
        # GUI layout
        grid = QGridLayout()
        self.setLayout(grid)

        self.autcorr = _pg.GraphicsLayoutWidget()
        self.autcorr.setAntialiasing(True)
        self.autcorr.acplot = self.autcorr.addPlot(row=0, col=0)
        self.xacplot = self.autcorr.acplot.plot(pen='r')
        self.yacplot = self.autcorr.acplot.plot(pen='g')

        # TODO: Wrap boilerplate into a function
        self.xyzGraph = _pg.GraphicsLayoutWidget()
        # self.xyzGraph.xPlot = self.xyzGraph.addPlot(row=0, col=0)

        # Data saving
        datagb = QGroupBox("Data")
        data_layout = QHBoxLayout()
        datagb.setLayout(data_layout)
        self.LoadDataButton = QPushButton("Load")
        data_layout.addWidget(self.LoadDataButton)
        # self.LoadDataButton.clicked.connect(lambda: self.goto_center())
        datagb.setFlat(True)

        # stats widget
        self.statWidget = QGroupBox("Live statistics")

        self.xstd_value = QLabel("-")
        self.ystd_value = QLabel("-")
        self.zstd_value = QLabel("-")

        stat_subgrid = QGridLayout()
        self.statWidget.setLayout(stat_subgrid)
        stat_subgrid.addWidget(QLabel("\u03C3X/nm"), 0, 0)
        stat_subgrid.addWidget(QLabel("\u03C3Y/nm"), 1, 0)
        stat_subgrid.addWidget(QLabel("\u03C3Z/nm"), 2, 0)
        stat_subgrid.addWidget(self.xstd_value, 0, 1)
        stat_subgrid.addWidget(self.ystd_value, 1, 1)
        stat_subgrid.addWidget(self.zstd_value, 2, 1)
        self.statWidget.setMinimumHeight(150)
        self.statWidget.setMinimumWidth(120)

        # drift and signal inensity graphs
        self.xyzGraph = _pg.GraphicsLayoutWidget()
        self.xyzGraph.setAntialiasing(True)

        # TODO: Wrap boilerplate into a function
        self.xyzGraph.xPlot = self.xyzGraph.addPlot(row=0, col=0)
        self.xyzGraph.xPlot.setLabels(bottom=("Time", "s"), left=("X shift", "nm"))
        self.xyzGraph.xPlot.showGrid(x=True, y=True)
        self.xmeanCurve = self.xyzGraph.xPlot.plot(pen="r", width=140)

        self.xyzGraph.yPlot = self.xyzGraph.addPlot(row=1, col=0)
        self.xyzGraph.yPlot.setLabels(bottom=("Time", "s"), left=("Y shift", "nm"))
        self.xyzGraph.yPlot.showGrid(x=True, y=True)
        self.ymeanCurve = self.xyzGraph.yPlot.plot(pen="r", width=140)

        self.xyzGraph.zPlot = self.xyzGraph.addPlot(row=2, col=0)
        self.xyzGraph.zPlot.setLabels(bottom=("Time", "s"), left=("Z shift", "nm"))
        self.xyzGraph.zPlot.showGrid(x=True, y=True)
        self.zCurve = self.xyzGraph.zPlot.plot(pen="y", connect="finite")

        # self.xyzGraph.avgIntPlot = self.xyzGraph.addPlot(row=3, col=0)
        # self.xyzGraph.avgIntPlot.setLabels(
        #     bottom=("Time", "s"), left=("Av. intensity", "Counts")
        # )
        # self.xyzGraph.avgIntPlot.showGrid(x=True, y=True)
        # self.avgIntCurve = self.xyzGraph.avgIntPlot.plot(pen="g")

        # xy drift graph (2D point plot)
        self.xyPoint = _pg.GraphicsLayoutWidget()
        self.xyPoint.resize(400, 400)
        self.xyPoint.setAntialiasing(False)

        self.xyplotItem = self.xyPoint.addPlot()
        self.xyplotItem.showGrid(x=True, y=True)
        self.xyplotItem.setLabels(
            bottom=("X position", "nm"), left=("Y position", "nm")
        )

        self.xyDataItem = self.xyplotItem.plot(
            [], pen=None, symbolBrush=(255, 0, 0), symbolSize=5, symbolPen=None
        )

        # z drift graph (1D histogram)
        x = _np.arange(-20, 20)
        y = _np.zeros(len(x))

        self.zHistogram = _pg.BarGraphItem(x=x, height=y, width=0.5, brush="#008a19")
        self.zPlot = self.xyPoint.addPlot()
        self.zPlot.addItem(self.zHistogram)

        # Lay everything in place
        grid.addWidget(self.autcorr, 0, 0, 1, 2)
        grid.addWidget(self.statWidget, 0, 2)
        grid.addWidget(self.xyzGraph, 1, 0)
        grid.addWidget(self.xyPoint, 1, 1, 1, 2)  # agrego 1,2 al final

    def closeEvent(self, *args, **kwargs):
        """Shut down stabilizer on exit."""
        super().closeEvent(*args, **kwargs)


if __name__ == "__main__":
    # Mock camera, replace with a real one

    if not QApplication.instance():
        app = QApplication([])
    else:
        app = QApplication.instance()
    gui = Frontend()

    gui.setWindowTitle("Drift Analysis")
    gui.show()
    gui.activateWindow()
    app.exec_()
    app.quit()
