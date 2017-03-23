# Rubidium Spectrum Model
This script provides python code that is useful for calculating absorption spectra for the Rubidium D1 and D2 lines.

Based on the paper by Paul Siddons:
*Siddons et al. J. Phys. B: At. Mol. Opt. Phys. 41, 155004 (2008).*
preprint available: [arXiv:0805.1139](http://arxiv.org/abs/0805.1139).

This code is also heavily inspired by Mathematica code by the same author. Unfortunately, the original code does not appear to be available any more.


## Features
Functions provide calculation of the following:
 - Transmission as a function of temperature (T) and detuning (∆).
 - Index of refraction *n(∆,T)*
 - Group velocity *V<sub>g</sub>(∆,T)*
 - Other intermediate parameters: vapor pressure, absorption, etc.

## Requirements (minimum tested)

 - python (2.7)
 - Scipy (0.9.0)
 - matplotlib 
 - numpy

I recommend using either [Anaconda](https://store.continuum.io/cshop/anaconda/) or [Enthought Canopy](https://store.enthought.com). These packages provide the above requirements in one easy-to-maintain format.

## Installation & Usage

The single-script python code `rubidiumD1.py` or `rubidiumD2.py` will generate absorption data for a vapor cell at a given temperature (in Kelvin) with length Lc (in meters). The detuning range generated can be set and defaults to -4 GHz through +6 GHz. The script generates a plot using pylab (matplotlib) and saves raw data to an ascii file.

An [IPython notebook](http://nbviewer.ipython.org/github/DawesLab/rubidium/blob/master/Rubidium%20Vapor.ipynb) is also included that provides an example workflow.

## Extending the code

Functions from this script can be imported and used in other python code. My intention is to continue to build on these features and maintain a useful python repository for numerical modeling in rubidium vapor systems.

Please contribute if you have more ideas and time. If this is useful, leave a comment and let me know what more you would like it to do.
