# Takyaq

Takyaq is a python module that stabilizes piezoelectric stages within few nm precision. It is is designed to be used in superresolution microscopy, altough it can be useful for many other applications, including photolithography.
The module accompanies and complements the article XXXX, which explains how it works and shows its performance.


## Features

 - Pure pyhthon implementation.
 - GUI agnostic.
 - Customizable and adaptable to any stage.
 - Tested and explained.
 
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

So far, the following extras are available:
 - PyQt example

Takyaq also includes a mock Piezo and Camera module for testing and development.
 
 
Takyaq uses a number of open source projects to work properly:

- [NumPy] - For efficient scientific computing
- [SciPy] - For data fitting

It can also profit of [Numba] for extra speed.

## What is included
 - A stabilizer module.
 - A fully functional PyQt frontend that is more than an example.
 - Mock camera and piezo modules, so you can develop, test and try without real equipment.
 - Different stabilization strategies so you can choose your own withouth rolling your own.
 
## How to use
### Module
Interfaces needed:
 - A camera module, that exposes a function named `get_XXX` and returns a single color
  image a (2D array)
 - A piezo module, that must expose two functions
   - One called `get_positions` that returns a 3 element collection (list, array or tuple) of x, y, and z positions *in nanometers*.
   - One called `set_positions` that takes 3 arguments: the x, y, and z positions where the stage should move *in nanometers*.

If the Python interfaces to your camera and stage use different naming or scale conventions, some adapting interfaces are provided in the XXX module.

Calibration data needed:
 - How many nanometers each camera pixel represents in the X and Y directions.
 - How the Z reflection spot moves when the Z position changes:
   - The direction (angle between the X axis and the line where the spot moves)
   - The number of pixels that the spot moves for each nanometers
 
The software provides a procedure to obtain the calibration data. Nevertheless, see below for calibration pitfalls.

### Stabilization strategies
The program comes with some predefined stabilization strategies:
  - PID
  - Short - time memory PID

You can implement your own strategies (for example ignoring fiduciary marks that have moved beyond a certain limit). Just see the code for some examples.
 
### PyQt frontend
The provided PyQt frontend is a fully functional example of how to use the stabilization module. For most purposes, you can use it _as is_.
 - Provide a camera (por ahora reemplazar `self._camera`)
 - Provide a piezo (por ahora reemplazar `self._piezo`)
 - Provide calibration values.
     - If you know the X and Y pixel size, introduce the values in the XXX global variables.
     - If you know the Z spot movement angle and displacement, introduce the values in the XXX variables.
     - If you do not know the values, just leave them as they are, and use the calibration procedure reported below
 - Run the program.
 - Manually focus the beads and Z spot reflection.
 - Create the Z ROI and move it to encompass the Z spot reflection.
 - You can check the _Z_ tracking checkbox right away.
 - If you have provided calibration values, you can also marck the _Z_ lock checkbox to stabilize the position.
 - Create as many _XY_ ROIs as you need. Move them to encompass one mark each.
 - You can check the _XY_ tracking checkbox.
 - If you have provided calibration values, you can also check the _XY_ lock checkbox to stabilize the position.

Calibration:
 - You need to have created the ROIs of the calibration you want to perform.
 - Be sure to have started tracking (but not locking).
 - Press the Desired "Calibrate" button. The calibration data will be printed to screen.
 - You might want to repeat and average the calibrations

### Camera and piezo orientation
  Blah blah

### Pixel sizes
Be careful since the pixel sizes used and determined using the provided calibration are only self-consistent. If the stage calibration is incorrect, the program will report valid but incorrect values for the stabilization precision.


## Some design comments

Porque los tiempos absolutos? Es lo mas conveniente para tener un mismo marco de referencia, salvo que la foto venga con esa informacion. En ese caso hay que hacer algo.

Run on a different thread. It could run on a different process for efficiency, but most applications need to be able to move the piezo from other modules. Managing the same stage from two different processes is usually much harder than from different threads on the same process.ç

## Some development comments
As we don't want to worry about formatting, we let [black] do its job.
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Citing

If you use this module or a derivation, please cite XXXX, and drop us a line.

## License

MIT???


   [SciPy]: <https://scipy.org/>
   [Poetry]: <http://angularjs.org>
   [NumPy]: <https://numpy.org/>
   [black]: <https://black.readthedocs.io/en/stable/>
   [Numba]: <https://numba.pydata.org/>
