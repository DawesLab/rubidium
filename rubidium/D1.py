#!/usr/bin/env python
# encoding: utf-8
"""
RubidiumD1.py

Software suite for calculating Rubidium D1 absorption spectra.

Example usage
    T = 273.15 + 35 # Temperature in Kelvin
    Lc = 0.075 # Length of cell in meters
    delta = linspace(-4,6,200)  # set up detuning values
    absdata = AbsorptionProfile(delta*1e9,T,Lc)  # calculate absorption
    plot(delta, absdata)  # plot absorption data 

This code is based on the paper by
Siddons et al. J. Phys. B: At. Mol. Opt. Phys. 41, 155004 (2008).

And heavily inspired by Mathematica code by the same author:
http://massey.dur.ac.uk/resources/resources.html
http://massey.dur.ac.uk/resources/psiddons/absdisD2.nb

Created by Andrew M. C. Dawes on 2015-07-15.
Copyright (c) 2015 Andrew Dawes. Some rights reserved.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>."""

import sys
import os
from scipy.constants import hbar, e, epsilon_0, c, m_u, k, m_e, alpha
from scipy import sqrt, pi, exp, zeros, array, real, imag
from scipy.special import erf
from numpy import savetxt, nan_to_num, roll, log10, seterr

seterr(invalid='ignore') # ignoring invalid warnings as the erf is touchy near ∆=0. No interesting physics there so we can safely ignore it.

a_0 = hbar/(m_e*c*alpha) # bohr radius

def u87(T):
    """Mean thermal velocity for 87-Rb
    expects T in Kelvin"""
    return sqrt(2*k*T/(86.909180520*m_u))

def u85(T):
    """Mean thermal velocity for 85-Rb
    expects T in Kelvin"""
    return sqrt(2*k*T/(84.911789732*m_u))


def P(T):
    """Vapor pressure in a thermal cell
    expects T in Kelvin"""
    # return 7e-8 # <--- USE THIS for finding background P
    if (T<312.46):
        return 10**( -94.04826 - 1961.258/T - 0.03771687*T + 42.57526*log10(T) )
    else:
        return 10**( 15.88253 - 4529.635/T + 0.00058663*T - 2.99138*log10(T) )


abundance = {"85":0.7217,"87":0.2783}

def N(T,isotope):
    """Atomic number density at T for given isotope
    expects T in Kelvin, and 
    isotope is either '85' or '87' """
    return abundance[isotope]*P(T)*133.323/(k*T)

lProbe = 794.978e-9

kProbe = 2*pi/lProbe

omegaProbe = kProbe*c

Gamma2 = 2*pi*5.746e6 # Excited state decay rate

d21 = sqrt(3)*sqrt(3*epsilon_0*hbar*Gamma2*lProbe**3/(8*pi**2)) # reduced matrix element

#####################################
# ELECTRICAL SUSCEPTIBILITY CONSTANTS
#####################################

def dispersion(a,y):
    """ Calculate dispersion profile for parameter a and frequency y"""
    disp = nan_to_num(1j/2.0*sqrt(pi)*(exp(1/4.0*(a-2j*y)**2)*(1-erf(a/2-1j*y))-exp(1/4.0*(a+2*1j*y)**2)*(1-erf(a/2.0+1j*y))))
    return disp

def voigt(a,y):
    """ Calculate Voigt profile for parameter a and frequency y"""
    try:
        voigt = nan_to_num(sqrt(pi)/2.0*(exp(1/4.0*(a-2j*y)**2)*((1-erf(a/2.0-1j*y))+exp(2*1j*a*y)*(1-erf(a/2.0+1j*y)))))
    except:
        pass # If this just passes, why try/except?

    return voigt

def lo87(T):
    """This is used as the parameter a in the voigt and dispersion calculations """
    return Gamma2/(kProbe*u87(T))

def D87(y, T):
    """ Caculate the dispersion for Rb-87 at frequency y and temperature T """
    return dispersion(lo87(T),y)

def V87(y, T):
    """ Calculate the voigt profile for Rb-87 at frequency y and temperature T """
    return voigt(lo87(T),y)

def lo85(T):
    """ This is used as the parameter a in the voigt and dispersion calculations """
    return Gamma2/(kProbe*u85(T))

def D85(y, T):
    """ Caculate the dispersion for Rb-85 at frequency y and temperature T """
    return dispersion(lo85(T),y)

def V85(y, T):
    """ Calculate the voigt profile for Rb-85 at frequency y and temperature T """
    return voigt(lo85(T),y)

# Transition strength factors C_F^2, call as F87[1][2] for the F=1 to F'=2 strength

F87 = [[ 0.0, 0.0, 0.0, 0.0],
       [ 0.0, 1/18.0, 5/18.0, 0.0],
       [ 0.0,  5/18.0, 5/18.0, 0.0]]

F85 = [[ 0.0, 0.0, 0.0, 0.0, 0.0],
       [ 0.0, 0.0, 0.0, 0.0, 0.0],
       [ 0.0, 0.0, 10/81.0, 35/81.0, 0.0],
       [ 0.0, 0.0,  35/81.0, 28/81.0, 0.0]]

# Detuning factors, call as det87[1][2] for the F=1 to F'=2 hyperfine transition, note the sign differs from Siddon's paper. Not sure why I did that? FIXME?

detF87 = [[ 0.0, 0.0, 0.0, 0.0],
          [ 0.0, -3820.046e6, -4632.339e6, 0.0],
          [ 0.0,  3014.644e6, 2202.381e6, 0.0]]

detF85 = [[ 0.0, 0.0, 0.0, 0.0, 0.0],
          [ 0.0, 0.0, 0.0, 0.0, 0.0],
          [ 0.0, 0.0, -1538.063e6, -1900.087e6, 0.0],
          [ 0.0, 0.0, 1497.657e6, 1135.721e6, 0.0]]

def K87(T,Fg,Fe):
    return F87[Fg][Fe]*1/8.0*1/(hbar*epsilon_0) * d21**2 * N(T, "87")/(kProbe * u87(T))

def K85(T,Fg,Fe):
    return F85[Fg][Fe]*1/12.0*1/(hbar*epsilon_0) * d21**2 * N(T, "85")/(kProbe * u85(T))

def chiRe87(delta,T,Fg,Fe):
    """The real part of the susceptibility for 87-Rb"""
    return K87(T, Fg,Fe)*real(D87(2*pi*(delta+detF87[Fg][Fe])/(kProbe*u87(T)), T))

def chiIm87(delta,T,Fg,Fe):
    """The imaginary part of the susceptibility for 87-Rb"""
    return K87(T, Fg, Fe)*real(V87(2*pi*(delta+detF87[Fg][Fe])/(kProbe*u87(T)), T))

def chiRe85(delta,T,Fg,Fe):
    """The real part of the susceptibility for 85-Rb"""
    return K85(T,Fg,Fe)*real(D85(2*pi*(delta+detF85[Fg][Fe])/(kProbe*u85(T)),T))

def chiIm85(delta,T,Fg,Fe):
    """The imaginary part of the susceptibility for 85-Rb"""
    return K85(T,Fg,Fe)*real(V85(2*pi*(delta+detF85[Fg][Fe])/(kProbe*u85(T)),T))

def TotalChiRe(delta,T):
    """The total real part of the susceptibility is the sum of each transition"""
    # Should clean this up with a decent loop or a map FIXME
    return chiRe87(delta,T,1,1) + chiRe87(delta,T,1,2) + chiRe87(delta,T,2,1) + chiRe87(delta,T,2,2) + chiRe85(delta,T,2,2) + chiRe85(delta,T,2,3) + chiRe85(delta,T,3,2) + chiRe85(delta,T,3,3)

def TotalChiIm(delta,T):
    """The total imaginary part of the susceptibility is the sum of each transition"""
    return chiIm87(delta,T,1,1) + chiIm87(delta,T,1,2) + chiIm87(delta,T,2,1) + chiIm87(delta,T,2,2) + chiIm85(delta,T,2,2) + chiIm85(delta,T,2,3) + chiIm85(delta,T,3,2) + chiIm85(delta,T,3,3)

def Totaln(delta,T):
    return 1+1/2.0*TotalChiRe(delta,T)

def TotalAlpha(delta,T):
    return kProbe*TotalChiIm(delta,T)

def Transmission(delta,T,Lc):
    """Transmission as a function of detuning, temperature, and cell length"""
    return exp(-TotalAlpha(delta,T)*Lc)

def groupVelocity(delta, T, Lc):
    ndata = Totaln(delta*1e9, T)
    dndw = (roll(ndata,-1) - ndata)/((delta[1] - delta[0])*1e9)  # numerical derivative dn/dw
    ng = ndata[:-1] + (delta[:-1]*1e9 + omegaProbe)*dndw[:-1]  # group index
    vg = 3e8/ng
    return vg


def transition_frequency87(Fg, Fe, Fe2=None):
    """
    Return the absolute frequency of the hyperfine transition Fg -> Fe for Rb87.

    Numbers from http://steck.us/alkalidata/

    :param Fg: Hyperfine quantum number F of the ground state (1,2)
    :param Fe: Hyperfine quantum number F of the excited state (1,2)
    :param Fe2: Optional parameter. If given, the cross-over transition (Fe, Fe2) is computed.
    :return: Transition frequency in Hz
    """

    # consistency check
    if not (isinstance(Fg, int) and isinstance(Fe, int)):
        raise TypeError

    if not ((1 <= Fg <= 2) and (1 <= Fe <= 2)):
        raise ValueError

    if Fe2 is not None:
        if not isinstance(Fe2, int):
            raise TypeError
        if not (1 <= Fe2 <= 2):
            raise ValueError

    freq = 377.107463380e12  # (S->P transition of Rb87 D1 line)

    freq_g = [0, 4.271676631815181e9, -2.563005979089109e9]
    freq_e = [0, -509.06e6, 305.44e6]

    if Fe2 is None:
        freq += freq_g[Fg] + freq_e[Fe]
    else:
        freq += freq_g[Fg] + (freq_e[Fe] + freq_e[Fe2])/2

    return freq


def transition_frequency85(Fg, Fe, Fe2=None):
    """
    Return the absolute frequency of the hyperfine transition Fg -> Fe for Rb85.

    Numbers from http://steck.us/alkalidata/

    :param Fg: Hyperfine quantum number F of the ground state (2,3)
    :param Fe: Hyperfine quantum number F of the excited state (2,3)
    :param Fe2: Optional parameter. If given, the cross-over transition (Fe, Fe2) is computed.
    :return: Transition frequency in Hz
    """

    # consistency check
    if not (isinstance(Fg, int) and isinstance(Fe, int)):
        raise TypeError

    if not ((2 <= Fg <= 3) and (2 <= Fe <= 3)):
        raise ValueError

    if Fe2 is not None:
        if not isinstance(Fe2, int):
            raise TypeError
        if not (2 <= Fe2 <= 3):
            raise ValueError

    freq = 377.107385690e12  # (S->P transition of Rb85 D1 line)

    freq_g = [0, 0, 1.7708439228e9, -1.2648885163e9]
    freq_e = [0, 0, -210.923e6, 150.659e6]

    if Fe2 is None:
        freq += freq_g[Fg] + freq_e[Fe]
    else:
        freq += freq_g[Fg] + (freq_e[Fe] + freq_e[Fe2])/2

    return freq


if __name__ == '__main__':
    from pylab import plot, show, linspace, title, xlabel, ylabel, subplot
    T = 273.15 + 30  # Temperature in Kelvin
    Lc = 0.075  # Length of cell in meters
    delta = linspace(-4, 6, 200)  # detuning in GHz
    transdata = Transmission(delta*1e9, T, Lc)
    ndata = Totaln(delta*1e9, T)

    vg = groupVelocity(delta,T,Lc)
    transit = Lc/vg
    timeshift = transit - Lc/c

    subplot(3, 1, 1)
    plot(delta, transdata)
    title("Rubidium D1 Spectrum at T= " + str(T) + " K")
    ylabel("Transmission (Arb. Units)")
    subplot(3, 1, 2)
    plot(delta, ndata)
    ylabel("Index n")
    subplot(3, 1, 3)
    plot(delta[:-1], timeshift*1e9)
    plot([delta[0], delta[-2]], [0, 0], color='#888888', linestyle='--', linewidth=1)
    xlabel(r"Detuning $\Delta$ (GHz)")
    ylabel(r"T (ns)")
    show()
    savetxt("D1AbsorptionData.dat", transdata)
