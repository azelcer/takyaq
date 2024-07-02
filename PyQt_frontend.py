# -*- coding: utf-8 -*-
"""
Sample PyQT frontend for Takyaq.

Uses a mocked camera and piezo motor: replace those parts with real interfaces to use.
Use:
    - Set the parameters:
         - nm per pixel XY
         - nm per pixel Z
    - Create XY ROIS and move and size them to encompass the fiducial marks
    - Create a Z ROI and move and size them to encompass the beam reflection
    - Start tracking of XY and Z rois. While tracking is active, erasing or changing
    the ROIs positions has no effect.
    - Start correction of XY and Z positions.

"""
import numpy as np
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
)

import pyqtgraph as pg

import logging as _lgn

from estabilizador import Stabilizer, MockCamera, MockPiezo, PointInfo, ROI

_lgr = _lgn.getLogger(__name__)


def pgROI2AZROI(roi: pg.ROI):
    x, y = roi.pos()
    w, h = roi.size()
    return ROI(x, x + w, y, y + h)


class GroupedCheckBoxes:
    """Manages grouped CheckBoxes states.

    This is a helper class to ease GUI implementation. It implements an 'All' checkbox
    that controls and stays sinchronized with other. It should be used carefully to
    avoid state changes loops.
    """

    def __init__(self, all_checkbox: QCheckBox, *other_checkboxes):
        """Init class.

        Parameters
        ----------
        all_checkbox : QCheckBox
            The checkbox that checks/unchecks all others.
        *other_checkboxes : TYPE
            All other checkboxes.
        """
        self.acb = all_checkbox
        self.others = other_checkboxes
        for _ in other_checkboxes:
            _.stateChanged.connect(self.on_state)
        all_checkbox.clicked.connect(self.on_click)

    def on_state(self, state: int):
        """Handle single items change."""
        self.acb.setChecked(all([_.isChecked() for _ in self.others]))

    def on_click(self, is_checked: bool):
        """Handle 'All' checkbox click."""
        for _ in self.others:
            _.setChecked(is_checked)


class QReader(QObject):
    """Helper class to send data from stabilizar to GUI.

    Implements a method that receives a PointInfo object. This is how the
    stabilizer thread reports the last data.

    In this case (a PyQT application), we emit a signal to the GUI.
    """

    new_data = pyqtSignal(float, np.ndarray, float, np.ndarray)

    def cb(self, data: PointInfo):
        """Report data."""
        self.new_data.emit(data.time, data.image, data.z_shift, data.xy_shifts)


# Physical parameters specific for each setup (should go into a configuration file)
_CAMERA_X_NMPPX = 23.5
_CAMERA_Y_NMPPX = 23.5
_CAMERA_Z_NMPPX = 36.0

# Globals (should go into a configuration file)
_MAX_POINTS = 200
_XY_ROI_SIZE = 60
_Z_ROI_SIZE = 100


class Frontend(QFrame):
    """PyQt Frontend for Takyq.

    Implemented as a QFrame so it can be easily integrated within a larger app.
    """

    _t_data = np.full((_MAX_POINTS,), np.nan)
    _z_data = np.full((_MAX_POINTS,), np.nan)
    _x_data = np.full((_MAX_POINTS, 0), np.nan)
    _counter = 0

    _x_plots = []
    _y_plots = []
    _roilist = []
    _z_ROI = None
    lastimage: np.ndarray = None
    _z_tracking_enabled: bool = False
    _xy_tracking_enabled: bool = False
    _z_locking_enabled: bool = False
    _xy_locking_enabled: bool = False

    def __init__(self, *args, **kwargs):
        """Init Frontend."""
        super().__init__(*args, **kwargs)

        self.setup_gui()
        # Mock camera, replace with a real one
        self._camera = MockCamera(
            _CAMERA_X_NMPPX,
            _CAMERA_Y_NMPPX,
            _CAMERA_Z_NMPPX,
            _CAMERA_X_NMPPX * 10,
            3,
            10,
        )
        # Mock piezo motor, replace with your own
        self._piezo = MockPiezo(self._camera)
        # Callback object
        self._cbojt = QReader()
        self._cbojt.new_data.connect(self.get_data)
        self.reset_data_buffers()
        self.reset_xy_data_buffers(len(self._roilist))
        self.reset_z_data_buffers()
        self._est = Stabilizer(
            self._camera, self._piezo, _CAMERA_X_NMPPX, _CAMERA_Z_NMPPX, self._cbojt.cb
        )
        self._est.start()

    def reset_data_buffers(self):
        """Reset data buffers unrelated to localization."""
        self._I_data = np.full((_MAX_POINTS,), np.nan)
        self._t_data = np.full((_MAX_POINTS,), np.nan)
        self._counter = 0

    def reset_xy_data_buffers(self, roi_len: int):
        """Reset data buffers related to XY localization."""
        self._x_data = np.full((_MAX_POINTS, roi_len), np.nan)  # sample #, roi
        self._y_data = np.full((_MAX_POINTS, roi_len), np.nan)  # sample #, roi

    def reset_z_data_buffers(self):
        """Reset data buffers related to Z localization."""
        self._z_data = np.full((_MAX_POINTS,), np.nan)

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

    @pyqtSlot(bool)
    def _add_xy_ROI(self, checked: bool):
        """Add a new XY ROI."""
        if self.lastimage is None:
            _lgr.warning("No image to set ROI")
            return
        w, h = self.lastimage.shape[0:2]
        ROIpos = (w / 2 - _XY_ROI_SIZE / 2, h / 2 - _XY_ROI_SIZE / 2)
        ROIsize = (_XY_ROI_SIZE, _XY_ROI_SIZE)
        roi = pg.ROI(ROIpos, ROIsize, rotatable=False)
        roi.addScaleHandle((1, 0), (0, 1), lockAspect=True)
        self.image_pi.addItem(roi)
        self._roilist.append(roi)
        self.delete_roiButton.setEnabled(True)

    @pyqtSlot(bool)
    def _remove_xy_ROI(self, checked: bool):
        """Remove last XY ROI."""
        if not self._roilist:
            _lgr.warning("No ROI to delete")
            return
        roi = self._roilist.pop()
        self.image_pi.removeItem(roi)
        del roi
        if not self._roilist:
            self.delete_roiButton.setEnabled(False)

    @pyqtSlot(bool)
    def _add_z_ROI(self, checked: bool):
        """Create the Z ROI."""
        if self._z_ROI:
            _lgr.warning("A Z ROI already exists")
            return
        w, h = self.lastimage.shape[0:2]
        ROIpos = (w / 2 - _XY_ROI_SIZE / 2, h / 2 - _XY_ROI_SIZE / 2)
        ROIsize = (_Z_ROI_SIZE, _Z_ROI_SIZE)
        roi = pg.ROI(ROIpos, ROIsize, pen={"color": "red", "width": 2}, rotatable=False)
        roi.addScaleHandle((1, 0), (0, 1), lockAspect=True)
        self.image_pi.addItem(roi)
        self._z_ROI = roi
        self.zROIButton.setEnabled(False)

    @pyqtSlot(int)
    def _send_z_rois(self, state: int):
        """Send Z roi data to the stabilizer and start tracking the Z position."""
        if state == Qt.CheckState.Unchecked:
            if self._z_locking_enabled:
                _lgr.warning("Z locking enabled: can not disable tracking")
                self.trackZBox.setCheckState(Qt.CheckState.Checked)
                return
            self._est.set_z_tracking(False)
            self._z_tracking_enabled = False
        else:
            if self._z_tracking_enabled:
                return
            if not self._z_ROI:
                _lgr.warning("We need a Z ROI to init tracking")
                self.trackZBox.setCheckState(Qt.CheckState.Unchecked)
                return
            self._est.set_z_roi(pgROI2AZROI(self._z_ROI))
            self.reset_z_data_buffers()
            if not self._xy_tracking_enabled:
                self.reset_data_buffers()
            self._est.set_z_tracking(True)
            self._z_tracking_enabled = True

    @pyqtSlot(int)
    def _start_z_lock(self, state: int):
        """Start stabilization of Z position."""
        if state == Qt.CheckState.Unchecked:
            self._est.set_z_stabilization(False)
            self._z_locking_enabled = False
        else:
            if not self._z_tracking_enabled:
                _lgr.warning("We need Z tracking to init locking")
                self.lockZBox.setCheckState(Qt.CheckState.Unchecked)
                return
            self._est.enable_z_stabilization()
            self._z_locking_enabled = True

    @pyqtSlot(int)
    def _send_xy_rois(self, state: int):
        """Send XY roi data to the stabilizer and start tracking the XY position."""
        if state == Qt.CheckState.Unchecked:
            if self._xy_locking_enabled:
                _lgr.warning("XY locking enabled: can not disable tracking")
                self.trackXYBox.setCheckState(Qt.CheckState.Checked)
                return
            self._est.set_xy_tracking(False)
            self._xy_tracking_enabled = False
        else:
            if not self._roilist:
                _lgr.warning("We need XY ROIs to init tracking")
                self.trackXYBox.setCheckState(Qt.CheckState.Unchecked)
                return
            if self._xy_tracking_enabled:
                return
            self.reset_graphs(len(self._roilist))
            self.reset_xy_data_buffers(len(self._roilist))
            if not self._z_tracking_enabled:
                self.reset_data_buffers()
            self._est.set_xy_rois([pgROI2AZROI(roi) for roi in self._roilist])
            self._est.set_xy_tracking(True)
            self._xy_tracking_enabled = True

    @pyqtSlot(int)
    def _start_xy_lock(self, state: int):
        """Start stabilization of XY position."""
        if state == Qt.CheckState.Unchecked:
            self._est.set_xy_stabilization(False)
            self._xy_locking_enabled = False
        else:
            if not self._xy_tracking_enabled:
                _lgr.warning("We need XY tracking to init locking")
                self.lockXYBox.setCheckState(Qt.CheckState.Unchecked)
                return
            self._est.enable_xy_stabilization()
            self._xy_locking_enabled = True

    @pyqtSlot(float, np.ndarray, float, np.ndarray)
    def get_data(self, t: float, img: np.ndarray, z: float, xy_shifts: np.ndarray):
        """Receive data from the stabilizer and graph it."""
        # manage image data
        self.img.setImage(img, autoLevels=self.lastimage is None)
        self.lastimage = img
        self._I_data[self._counter] = np.average(img)
        self._t_data[self._counter] = t
        t_data = self._t_data[: self._counter + 1]
        self.avgIntCurve.setData(t_data, self._I_data[: self._counter + 1])

        # manage tracking data
        if self._z_tracking_enabled:
            self._z_data[self._counter] = z
            # update reports
            self.zstd_value.setText(f"{np.nanstd(self._z_data[:self._counter]):.2f}")
            # update Graphs
            z_data = self._z_data[: self._counter + 1]
            self.zCurve.setData(t_data, z_data)
            hist, bin_edges = np.histogram(z_data, bins=30)
            self.zHistogram.setOpts(x=np.mean((bin_edges[:-1], bin_edges[1:],), axis=0),
                                    height=hist,
                                    width=bin_edges[1]-bin_edges[0])

        if self._xy_tracking_enabled and xy_shifts.shape[0]:
            self._x_data[self._counter] = xy_shifts[:, 0]
            self._y_data[self._counter] = xy_shifts[:, 1]
            # t_data = np.copy(t_data)

            x_data = self._x_data[: self._counter + 1]
            y_data = self._y_data[: self._counter + 1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_mean = np.nanmean(x_data, axis=1)
                y_mean = np.nanmean(y_data, axis=1)
                # update reports
                self.xstd_value.setText(
                    f"{np.nanstd(self._x_data[:self._counter + 1]):.2f}"
                )
                self.ystd_value.setText(
                    f"{np.nanstd(self._y_data[:self._counter + 1]):.2f}"
                )
            # update Graphs
            for i, p in enumerate(self._x_plots):
                p.setData(t_data, x_data[:, i])
            self.xmeanCurve.setData(t_data, x_mean)
            for i, p in enumerate(self._y_plots):
                p.setData(t_data, y_data[:, i])
            self.ymeanCurve.setData(t_data, y_mean)
            self.xyDataItem.setData(x_mean, y_mean)
        self._counter += 1

        # Meter esto al principio. Ver si grabar
        if self._counter >= _MAX_POINTS:  # roll o grabar
            self._t_data[0:-1] = self._t_data[1:]
            self._I_data[0:-1] = self._I_data[1:]
            if self._z_tracking_enabled:
                self._z_data[0:-1] = self._z_data[1:]
            if self._xy_tracking_enabled and xy_shifts.shape[0]:
                self._x_data[0:-1] = self._x_data[1:]
                self._y_data[0:-1] = self._y_data[1:]
                x_mean[0:-1] = x_mean[1:]
                y_mean[0:-1] = y_mean[1:]
            self._counter -= 1

    def setup_gui(self):
        """Create and lay out all GUI objects."""
        # GUI layout
        grid = QGridLayout()
        self.setLayout(grid)

        # image widget layout
        imageWidget = pg.GraphicsLayoutWidget()
        imageWidget.setMinimumHeight(250)
        imageWidget.setMinimumWidth(350)

        # setup axis
        self.xaxis = pg.AxisItem(orientation="bottom", maxTickLength=5)
        self.xaxis.showLabel(show=True)
        self.xaxis.setLabel("x", units="µm")

        self.yaxis = pg.AxisItem(orientation="left", maxTickLength=5)
        self.yaxis.showLabel(show=True)
        self.yaxis.setLabel("y", units="µm")
        self.xaxis.setScale(scale=_CAMERA_X_NMPPX / 1000)
        self.yaxis.setScale(scale=_CAMERA_Y_NMPPX / 1000)

        self.image_pi = imageWidget.addPlot(
            axisItems={"bottom": self.xaxis, "left": self.yaxis}
        )
        self.image_pi.setAspectLocked(True)
        self.img = pg.ImageItem()
        imageWidget.translate(-0.5, -0.5)
        self.image_pi.addItem(self.img)
        self.image_pi.setAspectLocked(True)
        imageWidget.setAspectLocked(True)

        self.hist = pg.HistogramLUTItem(image=self.img)
        self.hist.gradient.loadPreset("viridis")
        imageWidget.addItem(self.hist, row=0, col=1)

        # parameters widget
        self.paramWidget = QGroupBox("Tracking and feedback")
        self.paramWidget.setMinimumHeight(200)
        self.paramWidget.setFixedWidth(270)

        # ROI buttons
        self.xyROIButton = QPushButton("xy ROI")
        self.xyROIButton.clicked.connect(self._add_xy_ROI)
        self.zROIButton = QPushButton("z ROI")
        self.zROIButton.setEnabled(True)
        self.zROIButton.clicked.connect(self._add_z_ROI)
        self.delete_roiButton = QPushButton("Delete last xy ROI")
        self.delete_roiButton.clicked.connect(self._remove_xy_ROI)
        self.delete_roiButton.setEnabled(False)

        # Tracking control
        trackgb = QGroupBox("Tracking")
        trackLayout = QHBoxLayout()
        trackgb.setLayout(trackLayout)
        self.trackAllBox = QCheckBox("All")
        self.trackXYBox = QCheckBox("xy")
        self.trackZBox = QCheckBox("z")
        trackLayout.addWidget(self.trackAllBox)
        trackLayout.addWidget(self.trackXYBox)
        trackLayout.addWidget(self.trackZBox)
        trackgb.setMinimumSize(trackgb.sizeHint())

        self.trackManager = GroupedCheckBoxes(
            self.trackAllBox,
            self.trackXYBox,
            self.trackZBox,
        )
        self.trackZBox.stateChanged.connect(self._send_z_rois)
        self.trackXYBox.stateChanged.connect(self._send_xy_rois)

        # Correction controls
        lockgb = QGroupBox("Lock")
        lockLayout = QHBoxLayout()
        lockgb.setLayout(lockLayout)
        self.lockAllBox = QCheckBox("All")
        self.lockXYBox = QCheckBox("xy")
        self.lockZBox = QCheckBox("z")
        lockLayout.addWidget(self.lockAllBox)
        lockLayout.addWidget(self.lockXYBox)
        lockLayout.addWidget(self.lockZBox)
        lockgb.setMinimumSize(lockgb.sizeHint())
        self.lockManager = GroupedCheckBoxes(
            self.lockAllBox,
            self.lockXYBox,
            self.lockZBox,
        )
        self.lockZBox.stateChanged.connect(self._start_z_lock)
        self.lockXYBox.stateChanged.connect(self._start_xy_lock)

        self.exportDataButton = QPushButton("Export data")
        # self.clearDataButton = QPushButton('Clear data')
        subgrid = QGridLayout()
        self.paramWidget.setLayout(subgrid)

        subgrid.addWidget(self.xyROIButton, 0, 0)
        subgrid.addWidget(self.zROIButton, 1, 0)
        subgrid.addWidget(self.delete_roiButton, 2, 0)

        subgrid.addWidget(trackgb, 4, 0)
        subgrid.addWidget(lockgb, 5, 0)
        subgrid.addWidget(self.exportDataButton, 6, 0)
        # subgrid.addWidget(self.clearDataButton, 7, 0)

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
        self.xyzGraph = pg.GraphicsLayoutWidget()
        self.xyzGraph.setAntialiasing(True)

        # TODO: HAcer funci'on 'unica de creacion
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
        self.zCurve = self.xyzGraph.zPlot.plot(pen="y")

        self.xyzGraph.avgIntPlot = self.xyzGraph.addPlot(row=3, col=0)
        self.xyzGraph.avgIntPlot.setLabels(
            bottom=("Time", "s"), left=("Av. intensity", "Counts")
        )
        self.xyzGraph.avgIntPlot.showGrid(x=True, y=True)
        self.avgIntCurve = self.xyzGraph.avgIntPlot.plot(pen="g")

        # xy drift graph (2D point plot)
        self.xyPoint = pg.GraphicsLayoutWidget()
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
        x = np.arange(-20, 20)
        y = np.zeros(len(x))

        self.zHistogram = pg.BarGraphItem(x=x, height=y, width=0.5, brush="#008a19")
        self.zPlot = self.xyPoint.addPlot()
        self.zPlot.addItem(self.zHistogram)

        # Lay everything in place
        grid.addWidget(imageWidget, 0, 0)
        grid.addWidget(self.paramWidget, 0, 1)
        grid.addWidget(self.statWidget, 0, 2)
        grid.addWidget(self.xyzGraph, 1, 0)
        grid.addWidget(self.xyPoint, 1, 1, 1, 2)  # agrego 1,2 al final

    def closeEvent(self, *args, **kwargs):
        """Shut down stabilizer on exit."""
        super().closeEvent(*args, **kwargs)
        self._est.stop()


if __name__ == "__main__":
    if not QApplication.instance():
        app = QApplication([])
    else:
        app = QApplication.instance()
    gui = Frontend()

    gui.setWindowTitle("Takyq with PyQt frontend")
    gui.show()
    app.exec_()
    app.quit()
