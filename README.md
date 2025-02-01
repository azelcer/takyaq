# Takyaq

Takyaq is a python module that stabilizes piezoelectric stages within few nm precision. It is is designed to be used in superresolution microscopy, altough it can be useful for many other applications, including photolithography.
The module accompanies and complements the article XXXX, which explains how it works and shows its performance.


## Features

 - Pure pyhthon implementation.
 - GUI agnostic.
 - Customizable and adaptable to any stage.
 - Tested and explained.


## What is included
 - A stabilizer module.
 - A fully functional PyQt frontend that is more than an example.
 - Mock camera and piezo modules, so you can develop, test and try without real equipment.
 - Different stabilization strategies so you can choose your own withouth rolling your own.
 
## How to install
Takyaq is not yet available in PyPi. Meanwhile installation and development is managed using [Poetry].
Clone the repository, either using `ssh`:
```sh
git clone git@github.com:azelcer/takyaq.git
```
or `https`:
```sh
git clone https://github.com/azelcer/pyflux.git
```

Install required dependencies using:
```sh
cd takyaq
poetry install
```

If you want to add qt examples, you can use:
```sh
poetry install --with qt
```

Takyaq also includes a mock Piezo and Camera module for testing and development.
 

Takyaq uses a number of open source projects to work properly:

- [NumPy] - For efficient scientific computing
- [SciPy] - For data fitting

It can also profit of [Numba] for extra speed.

## How to use
### Module
Interfaces needed:
 - A camera module, that exposes a function named `get_XXX` and returns a single color
  image (a [NumPy] 2D array)
 - A piezo module, that must expose three functions
   - One called `get_position` that returns a 3-element collection (list, array, tuple, etc) of x, y, and z positions *in nanometers*.
   - One called `set_position_xy` that takes 2 arguments: the x and y positions where the stage should move *in nanometers*.
   - One called `set_position_z` that takes 1 arguments: the z position where the stage should move *in nanometers*.

If the Python interfaces to your camera and stage use different naming conventions or units, you should write some wrapping classes (see below).

Calibration data needed:
 - How many nanometers each camera pixel represents in the X and Y directions.
 - How the Z reflection spot moves when the Z position changes:
   - The direction (angle between the X axis and the line where the spot moves)
   - The number of pixels that the spot moves for each nanometers
 
The software provides a procedure to obtain the calibration data. Nevertheless, see below for calibration pitfalls.

The module communicates with other modules using callbacks. The callback procedure should do its job (put the data in a queue, etc.) fast, so the latency between position corrections is short. Check the PyQt example for a basic idea.

### Stabilization strategies
The program comes with some predefined stabilization strategies:
  - PI

You can implement your own strategies (for example ignoring fiduciary marks that have moved beyond a certain limit). Just see the `controllers` module for some examples.

### Adapting cameras and piezos.
A piezo and a camera Abstract Base Class (ABC) are provided in the `base_classes` module. Use them to ensure that the method signatures are adecuate.

Assume you have a piezo controller API that exposes functions that are called `move_to` that takes as parameter the axis as a string (`'X'`, `'Y'` or `'Z'`) and the new position in µm, and `get_pos` instead of `get_position`, that returns the values in micrometers. You can use a wrapper like this one:
```python
import my_piezo_driver
from takyaq.base_classes import BasePiezo

class WrappedPiezo(Baseiezo):
   def __init__(self):
       # Do whatever your driver needs you to do
       self._motor = my_piezodriver.get_motor()
       self._motor.init()

    def close(self):
        self._motor.init()

    def set_position_xy(self, x: float, y: float):
        self._motor.move_to('X', x / 1E3)
        self._motor.move_to('Y', y / 1E3)

    def set_position_z(self, z: float):
        self._motor.move_to('Z', z / 1E3)

    def get_position(self) -> tuple[float, float, float]:
        return tuple(p * 1E3 for p in self._motor.get_pos())


piezo = WrappedPiezo()     
# Use `piezo` with takyaq 
...

piezo.close()


```
Making a context manager is simple and very convenient.

### PyQt frontend
The provided PyQt frontend is a fully functional example of how to use the stabilization module. For most purposes, you can use it _as is_. You must provide:
 - A camera object.
 - A piezo object.
 - Calibration values (as a `info_types.CameraInfo` object). If you do not know these values, just use a value of 1 for each one, and perform the calibration procedure reported below.

Once you run the program, perform the following steps:
 - Press the `Show options window` button and set very low `Kp` value for all axis, and set all `Ki` to 0
 - Manually find the focus.
 - Create the Z ROI and move it to encompass the Z spot reflection.
 - You can check the _Z_ tracking checkbox right away.
 - If you have provided calibration values, you can also mark the _Z_ lock checkbox to stabilize the focus.
 - Create as many _XY_ ROIs as you feel. Move and adjust their size to encompass one fiducial mark each.
 - You can check the _XY_ tracking checkbox.
 - If you have provided calibration values, you can also check the _XY_ lock checkbox to stabilize the position.

Calibration:
 - You need to have created the ROIs of the calibration you want to perform (XY or Z).
 - Be sure to have started tracking (but not locking).
 - Open the options window and press the Desired "Calibrate" button. The calibration data will be logged to screen. A nicer report will be implemented on the near future.
 - You might want to repeat and average the calibrations

### Camera and piezo orientation
  Depending on the assembly, the camera X-Y axis can be switched, or one of them flipped. Correct this in the `camera` wrapper 

### Pixel sizes
Be careful since the pixel sizes used and determined using the provided calibration are only self-consistent. If the stage calibration is incorrect, the program will report valid but incorrect values for the stabilization precision.


## Some design comments and pitfalls
 - We try to keep compatibility with Python 3.7+, to support legacy setups.
 - Times are reported in seconds since the Epoch, as provided by Python's `time.time()`, since it is the most convenient (not the best) way of having the same time reference between different programs.
 - The stabilization loop runs on a different thread. It could run on a different process for efficiency (to avoid GIL issues), but most applications need to be able to move the piezo from other modules. Managing the same stage from two different processes is usually much harder than from different threads on the same process.
 - Fitting of the fiducial beads is done in multiple processes to achieve higer correction frequencies. This implies that the many processes are spawned. Objects created on module load are created on each process and might collide (open ports, etc.).
 - More information can be extracted from the calibrations (like coupling between axes). This kinf on analyses might be implemented in the future.

## Some development comments
As we don't want to worry about formatting, fro time to time we let [black] do its job.
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Citing

If you use this module or a derivation, please cite XXXX, and drop us a line.

## License

MIT???


   [SciPy]: <https://scipy.org/>
   [Poetry]: <http://angularjs.org>
   [NumPy]: <https://numpy.org/>
   [black]: <https://black.readthedocs.io/en/stable/>
