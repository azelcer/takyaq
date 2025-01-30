# -*- coding: utf-8 -*-
"""
Sample PyQT frontend for Takyaq.

Uses a mocked camera and piezo motor: replace those parts with real interfaces
to have a fully functional stabilization program.

Use:
    - Set the parameters:
         - nm per pixel XY
         - nm per pixel Z
    - Create XY ROIS and move and size them to encompass the fiducial marks
    - Create a Z ROI and move and size them to encompass the beam reflection
    - Start tracking of XY and Z rois. While tracking is active, erasing or
    changing the ROIs positions has no effect.
    - Start correction of XY and Z positions.

"""
import logging as _lgn

from PyQt5.QtWidgets import QApplication
from takyaq.frontends.PyQt_frontend import Frontend
from takyaq.stabilizer import Stabilizer
from takyaq.controllers import PIController
from takyaq.mocks import MockCamera, MockPiezo
from takyaq.info_types import CameraInfo


_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


# Constants for camera mock
_CAMERA_XY_NMPPX = 23.5
_CAMERA_Z_NMPPX = 10
_CAMERA_Z_ROTATION = 3.1415 / 4


if __name__ == "__main__":
    camera_info = CameraInfo(
        _CAMERA_XY_NMPPX,
        _CAMERA_Z_NMPPX,
        _CAMERA_Z_ROTATION,
    )
    # Mock camera, replace with a real one
    camera = MockCamera(
        _CAMERA_XY_NMPPX,
        _CAMERA_XY_NMPPX,
        _CAMERA_Z_NMPPX,
        _CAMERA_XY_NMPPX * 7,  # en pixeles
        _CAMERA_Z_ROTATION,
        1,  # Center position noise in pixels
        10,
    )
    # Mock piezo motor, replace with your own
    piezo = MockPiezo(camera)
    controller = PIController()

    if not QApplication.instance():
        app = QApplication([])
    else:
        app = QApplication.instance()
    app.setStyle('Windows')
    with Stabilizer(camera, piezo, camera_info, controller) as stb:
        gui = Frontend(camera, piezo, controller, camera_info, stb)
        gui.setWindowTitle("Takyaq with PyQt frontend")
        gui.show()
        gui.raise_()
        gui.activateWindow()

        app.exec_()
        app.quit()
