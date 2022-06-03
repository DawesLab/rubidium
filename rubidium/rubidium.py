#!/usr/bin/env python
# encoding: utf-8

"""
rubidium.py

Software suite for calculating Rubidium absorption spectra.

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

You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""


from scipy.constants import hbar, e, epsilon_0, c, m_u, k, m_e, alpha
from scipy import sqrt, pi, exp, zeros, array, real, imag
from scipy.special import erf
from pylab import plot, show, linspace, title, xlabel, ylabel, subplot
from numpy import savetxt, nan_to_num, roll, log10, seterr

seterr(invalid='ignore') # ignoring invalid warnings as the erf is touchy near ∆=0. No interesting physics there so we can safely ignore it.

a_0 = hbar/(m_e*c*alpha) # bohr radius


def dispersion(a, y):
    """ Calculate dispersion profile for parameter a and frequency y"""
    disp = nan_to_num(1j/2.0*sqrt(pi)*(exp(1/4.0*(a-2j*y)**2)*(1-erf(a/2-1j*y))-exp(1/4.0*(a+2*1j*y)**2)*(1-erf(a/2.0+1j*y))))
    return disp


def voigt(a, y):
    """ Calculate Voigt profile for parameter a and frequency y"""
    v = nan_to_num(sqrt(pi)/2.0*(exp(1/4.0*(a-2j*y)**2)*((1-erf(a/2.0-1j*y))+exp(2*1j*a*y)*(1-erf(a/2.0+1j*y)))))
    return v


class Rubidium:
    """
    Rubidium class.

    Attributes:
    :param int line: Set the D-line. 1: D1, 2: D2. Defaults to 1.
    :param int abundance: Abundance of Rb-85. Defaults to 0.7217 (natural abundance).
        Abundance of Rb-87 is 1-abundance.

    """
    def __init__(self, line=1, abundance=0.7217):
        # === private ===
        self._lProbe = 794.978e-9
        self._kProbe = 2*pi/self._lProbe
        self._omegaProbe = self._kProbe*c
        self._Gamma2 = 2*pi*5.746e6  # Excited state decay rate
        self._d21 = sqrt(3)*sqrt(3*epsilon_0*hbar*self._Gamma2*self._lProbe**3/(8*pi**2))  # reduced matrix element

        # Transition strength factors C_F^2, call as F87[1][2] for the F=1 to F'=2 strength
        self._F87 = None
        self._F85 = None

        # Detuning factors, call as det87[1][2] for the F=1 to F'=2 hyperfine transition.
        self._detF87 = None
        self._detF85 = None

        # === public (see setter) ===
        self._line = self.line = line
        self._abundance = self.abundance = abundance

    @property
    def line(self):
        return self._line

    @line.setter
    def line(self, val):
        if val == 1 or val == 2:
            self._line = val
        else:
            raise ValueError("Line can only be 1 (D1) or 2 (D2).")

        if val == 1:
            self._lProbe = 794.978e-9
            self._Gamma2 = 2*pi*5.746e6
            self._F87 = [[0.0, 0.0, 0.0, 0.0],
                         [0.0, 1/18.0, 5/18.0, 0.0],
                         [0.0, 5/18.0, 5/18.0, 0.0]]
            self._F85 = [[0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 10/81.0, 35/81.0, 0.0],
                         [0.0, 0.0,  35/81.0, 28/81.0, 0.0]]

            # Detuning factors, call as det87[1][2] for the F=1 to F'=2 hyperfine transition,
            # note the sign differs from Siddon's paper. Not sure why I did that? FIXME?
            self._detF87 = [[0.0, 0.0, 0.0, 0.0],
                            [0.0, -3820.046e6, -4632.339e6, 0.0],
                            [0.0,  3014.644e6, 2202.381e6, 0.0]]

            self._detF85 = [[0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, -1538.063e6, -1900.087e6, 0.0],
                            [0.0, 0.0, 1497.657e6, 1135.721e6, 0.0]]

        if val == 2:
            self._lProbe = 780.241e-9
            self._Gamma2 = 2*pi*6.065e6
            self._F87 = [[0.0, 0.0, 0.0, 0.0],
                         [1/9.0, 5/18.0, 5/18.0, 0.0],
                         [0.0,  1/18.0, 5/18.0, 7/9.0]]
            self._F85 = [[0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 1/3.0, 35/81.0, 28/81.0, 0.0],
                         [0.0, 0.0,  10/81.0, 35/81.0, 1.0]]
            self._detF87 = [[0.0, 0.0, 0.0, 0.0],
                            [-4027.403e6, -4099.625e6, -4256.570e6, 0.0],
                            [0.0,  2735.050e6, 2578.110e6, 2311.260e6]]

            self._detF85 = [[0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, -1635.454e6, -1664.714e6, -1728.134e6, 0.0],
                            [0.0, 0.0, 1371.290e6, 1307.870e6, 1186.910e6]]

        self._kProbe = 2*pi/self._lProbe
        self._omegaProbe = self._kProbe*c
        self._d21 = sqrt(3)*sqrt(3*epsilon_0*hbar*self._Gamma2*self._lProbe**3/(8*pi**2))

    @property
    def abundance(self):
        return self._abundance

    @abundance.setter
    def abundance(self, val):
        if 0 <= val <= 1:
            self._abundance = val
        else:
            raise ValueError("Abundance must be between 0 and 1.")

    @staticmethod
    def u87(T):
        """
        Mean thermal velocity for 87-Rb

        :param float T: Temperature in K
        :return: Mean thermal velocity
        """
        return sqrt(2*k*T/(86.909180520*m_u))

    @staticmethod
    def u85(T):
        """
        Mean thermal velocity for 85-Rb

        :param float T: Temperature in K
        :return: Mean thermal velocity
        """
        return sqrt(2*k*T/(84.911789732*m_u))

    @staticmethod
    def P(T):
        """
        Vapor pressure in a thermal cell

        :param float T: Temperature in K
        :return: Vapor pressure in Pa
        """
        # return 7e-8 # <--- USE THIS for finding background P
        if T < 312.46:
            return 10**(-94.04826 - 1961.258/T - 0.03771687*T + 42.57526*log10(T))
        else:
            return 10**(15.88253 - 4529.635/T + 0.00058663*T - 2.99138*log10(T))

    def N(self, T, isotope):
        """
        Atomic number density at T for given isotope
        :param float T: Temperature in K
        :param str isotope: either '85' or '87'
        :return: Density (in 1/m**3)
        """
        if isotope == "85":
            abundance = self.abundance
        elif isotope == "87":
            abundance = 1 - self.abundance

        return abundance*self.P(T)*133.323/(k*T)

    def lo87(self, T):
        """
        This is used as the parameter a in the voigt and dispersion calculations

        :param float T: Temperature in K
        """
        return self._Gamma2/(self._kProbe*self.u87(T))

    def D87(self, y, T):
        """
        Calculate the dispersion for Rb-87 at frequency y and temperature T

        :param float y: Frequency
        :param float T: Temperature in K
        """
        return dispersion(self.lo87(T), y)

    def V87(self, y, T):
        """
        Calculate the voigt profile for Rb-87 at frequency y and temperature T
        :param float y: Frequency
        :param float T: Temperature in K
        """
        return voigt(self.lo87(T), y)

    def lo85(self, T):
        """ This is used as the parameter a in the voigt and dispersion calculations """
        return self._Gamma2/(self._kProbe*self.u85(T))

    def D85(self, y, T):
        """
        Calculate the dispersion for Rb-85 at frequency y and temperature T
        :param float y: Frequency
        :param float T: Temperature in K
        """
        return dispersion(self.lo85(T), y)

    def V85(self, y, T):
        """
        Calculate the voigt profile for Rb-85 at frequency y and temperature T
        :param float y: Frequency
        :param float T: Temperature in K
        """
        return voigt(self.lo85(T), y)

    def K87(self, T, Fg, Fe):
        """

        :param T: Temperature in K
        :param Fg: Hyperfine quantum number of ground state
        :param Fe: Hyperfine quantum number of excited state
        :return:
        """
        return self._F87[Fg][Fe]*1/8.0*1/(hbar*epsilon_0) * self._d21**2 * self.N(T, "87")/(self._kProbe*self.u87(T))

    def K85(self, T, Fg, Fe):
        """

        :param T: Temperature in K
        :param Fg: Hyperfine quantum number of ground state
        :param Fe: Hyperfine quantum number of excited state
        :return:
        """
        return self._F85[Fg][Fe]*1/12.0*1/(hbar*epsilon_0) * self._d21**2 * self.N(T, "85")/(self._kProbe * self.u85(T))

    def chiRe87(self, delta, T, Fg, Fe):
        """
        The real part of the susceptibility for 87-Rb

        :param float delta: Relative frequency detuning
        :param float T: Temperature in K
        :param int Fg: Hyperfine quantum number of ground state
        :param int Fe: Hyperfine quantum number of excited state
        :return: The real part of the susceptibility for 87-Rb
        """
        return self.K87(T, Fg, Fe)*real(self.D87(2*pi*(delta+self._detF87[Fg][Fe])/(self._kProbe*self.u87(T)), T))

    def chiIm87(self, delta, T, Fg, Fe):
        """
        The imaginary part of the susceptibility for 87-Rb

        :param float delta: Relative frequency detuning
        :param float T: Temperature in K
        :param int Fg: Hyperfine quantum number of ground state
        :param int Fe: Hyperfine quantum number of excited state
        :return: The imaginary part of the susceptibility for 87-Rb
        """
        return self.K87(T, Fg, Fe)*real(self.V87(2*pi*(delta+self._detF87[Fg][Fe])/(self._kProbe*self.u87(T)), T))

    def chiRe85(self, delta, T, Fg, Fe):
        """
        The real part of the susceptibility for 85-Rb

        :param float delta: Relative frequency detuning
        :param float T: Temperature in K
        :param int Fg: Hyperfine quantum number of ground state
        :param int Fe: Hyperfine quantum number of excited state
        :return: The real part of the susceptibility for 85-Rb
        """
        return self.K85(T, Fg, Fe)*real(self.D85(2*pi*(delta+self._detF85[Fg][Fe])/(self._kProbe*self.u85(T)), T))

    def chiIm85(self, delta, T, Fg, Fe):
        """
        The imaginary part of the susceptibility for 85-Rb

        :param float delta: Relative frequency detuning
        :param float T: Temperature in K
        :param int Fg: Hyperfine quantum number of ground state
        :param int Fe: Hyperfine quantum number of excited state
        :return: The imaginary part of the susceptibility for 85-Rb
        """
        return self.K85(T, Fg, Fe)*real(self.V85(2*pi*(delta+self._detF85[Fg][Fe])/(self._kProbe*self.u85(T)), T))

    def TotalChi(self, delta, T):
        """
        The total real and imaginary part of the susceptibility is the sum of each transition

        :param float delta: Relative frequency detuning
        :param float T: Temperature in K
        :return: total real part of the susceptibility, total imaginary part of the susceptibility

        """
        total_real = 0
        total_imag = 0

        if self.line == 1:  # D1 line
            for idx_g in range(1, 3):  # sum over all Rb-87 transitions (Fg=1,2; Fe=1,2)
                for idx_e in range(1, 3):
                    total_real += self.chiRe87(delta, T, idx_g, idx_e)
                    total_imag += self.chiIm87(delta, T, idx_g, idx_e)

            for idx_g in range(2, 4): # sum over all Rb-85 transitions (Fg=2,3; Fe=2,3)
                for idx_e in range(2, 4):
                    total_real += self.chiRe85(delta, T, idx_g, idx_e)
                    total_imag += self.chiIm85(delta, T, idx_g, idx_e)

        elif self.line == 2:  # D2 line
            for idx_g in range(1, 3):  # sum over all Rb-87 transitions (Fg=1,2; Fe=0,1,2,3; Delta_F=1)
                for idx_e in range(-1, 2):
                    total_real += self.chiRe87(delta, T, idx_g, idx_e + idx_g)
                    total_imag += self.chiIm87(delta, T, idx_g, idx_e + idx_g)

            for idx_g in range(2, 4):  # sum over all Rb-85 transitions (Fg=2,3; Fe=1,2,3,4; Delta_F=1)
                for idx_e in range(-1, 2):
                    total_real += self.chiRe85(delta, T, idx_g, idx_e + idx_g)
                    total_imag += self.chiIm85(delta, T, idx_g, idx_e + idx_g)

        return total_real, total_imag

    def Totaln(self, delta, T):
        """
        Calculate refractive index

        :param float delta: Relative frequency detuning
        :param float T: Temperature in K
        :return: Refractive index
        """
        total_chi_real, _ = self.TotalChi(delta, T)
        return 1+1/2.0*total_chi_real

    def TotalAlpha(self, delta, T):
        """
        Calculate absorption coefficient

        :param float delta: Relative frequency detuning
        :param float T: Temperature in K
        :return: Absorption coefficient
        """
        _, total_chi_imag = self.TotalChi(delta, T)
        return self._kProbe*total_chi_imag

    def Transmission(self, delta, T, Lc):
        """
        Transmission as a function of detuning, temperature, and cell length

        :param delta: Relative frequency detuning
        :param T: Temperature in K
        :param Lc: Cell length
        :return: Transmission through cell
        """
        return exp(-self.TotalAlpha(delta, T)*Lc)

    def groupVelocity(self, delta, T):
        """

        :param float delta: Relative frequency detuning
        :param float T: Temperature in K
        :return: Group velocity
        """
        ndata = self.Totaln(delta*1e9, T)
        dndw = (roll(ndata, -1) - ndata)/((delta[1] - delta[0])*1e9)  # numerical derivative dn/dw
        ng = ndata[:-1] + (delta[:-1]*1e9 + self._omegaProbe)*dndw[:-1]  # group index
        vg = 3e8/ng
        return vg

    def transition_frequency87(self, Fg, Fe, Fe2=None):
        """
        Return the absolute frequency of the hyperfine transition Fg -> Fe for Rb87.

        Numbers from http://steck.us/alkalidata/

        :param Fg: Hyperfine quantum number F of the ground state (D1: 1,2; D2: 1,2)
        :param Fe: Hyperfine quantum number F of the excited state (D1: 1,2; D2: 0,1,2,3)
        :param Fe2: Optional parameter. If given, the cross-over transition (Fe, Fe2) is computed.
        :return: Transition frequency in Hz
        """
        freq = None

        if self.line == 1:  # D1 line
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

        if self.line == 2:  # D2 line
            # consistency check
            if not (isinstance(Fg, int) and isinstance(Fe, int)):
                raise TypeError

            if not ((1 <= Fg <= 2) and (0 <= Fe <= 3)):
                raise ValueError

            if Fe2 is not None:
                if not isinstance(Fe2, int):
                    raise TypeError
                if not (0 <= Fe2 <= 3):
                    raise ValueError

            freq = 384.2304844685e12  # (S->P transition of Rb87 D2 line)

            freq_g = [0, 4.271676631815181e9, -2.563005979089109e9]
            freq_e = [-302.0738e6, -229.8518e6, -72.9112e6, 193.7407e6]

            if Fe2 is None:
                freq += freq_g[Fg] + freq_e[Fe]
            else:
                freq += freq_g[Fg] + (freq_e[Fe] + freq_e[Fe2])/2

        return freq

    def transition_frequency85(self, Fg, Fe, Fe2=None):
        """
        Return the absolute frequency of the hyperfine transition Fg -> Fe for Rb85.

        Numbers from http://steck.us/alkalidata/

        :param Fg: Hyperfine quantum number F of the ground state (D1: 2,3; D2: 2,3)
        :param Fe: Hyperfine quantum number F of the excited state (D1: 2,3; 1,2,3,4)
        :param Fe2: Optional parameter. If given, the cross-over transition (Fe, Fe2) is computed.
        :return: Transition frequency in Hz
        """
        freq = None

        if self.line == 1:  # D1 line
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

        if self.line == 2:  # D2 line
            # consistency check
            if not (isinstance(Fg, int) and isinstance(Fe, int)):
                raise TypeError

            if not ((2 <= Fg <= 3) and (1 <= Fe <= 4)):
                raise ValueError

            if Fe2 is not None:
                if not isinstance(Fe2, int):
                    raise TypeError
                if not (1 <= Fe2 <= 4):
                    raise ValueError

            freq = 384.230406373e12  # (S->P transition of Rb85 D2 line)

            freq_g = [0, 0, 1.7708439228e9, -1.2648885163e9]
            freq_e = [0, -113.208e6, -83.835e6, -20.435e6, 100.205e6]

            if Fe2 is None:
                freq += freq_g[Fg] + freq_e[Fe]
            else:
                freq += freq_g[Fg] + (freq_e[Fe] + freq_e[Fe2])/2

        return freq

if __name__ == '__main__':
    # ==================== D1 =====================
    rb = Rubidium(line=1)
    T = 273.15 + 30  # Temperature in Kelvin
    Lc = 0.075  # Length of cell in meters
    delta = linspace(-4, 6, 200)  # detuning in GHz
    transdata = rb.Transmission(delta*1e9, T, Lc)
    ndata = rb.Totaln(delta*1e9, T)

    vg = rb.groupVelocity(delta,T)
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
    savetxt("D1AbsorptionData2.dat", transdata)

    # ==================== D2 =====================
    rb = Rubidium(line=2)
    T = 273.15 + 47  # Temperature in Kelvin
    Lc = 0.075  # Length of cell in meters
    delta = linspace(-4, 6, 200)  # detuning in GHz
    transdata = rb.Transmission(delta*1e9, T, Lc)
    ndata = rb.Totaln(delta*1e9, T)

    vg = rb.groupVelocity(delta,T)
    transit = Lc/vg
    timeshift = transit - Lc/c

    subplot(3, 1, 1)
    plot(delta, transdata)
    title("Rubidium D2 Spectrum at T= " + str(T) + " K")
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
    savetxt("D2AbsorptionData2.dat", transdata)