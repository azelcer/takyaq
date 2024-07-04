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
Clone the repository:
```sh
git clone XXXXXXXX
```
and then install required depenencies using:
```sh
poetry install
```
If you want to add some examples, you can use:
```sh
Ver cómo instalar los extras
```

So far, the following extras are available:
 - PyQt example
 - Mock Piezo and Camera modules
 
 
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
Blah Blah

### Camera and piezo orientation

Be careful


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
