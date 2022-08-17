#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a Python implementation of modal identification on Morlet-
wave method, based on [1]_:

.. [1] I. Tomac, J. Slavič, Modal Identification using Morlet-Wave,
Mechanical Systems and Signal Processing. xx (202x) xx–xx.
_doi: 10.1016/j.ymssp.20xx.xx.xx.
.. _doi: https://doi.org/10.1016/j.ymssp.20xx.xx.xx

Check web page of EU project `NOSTRADAMUS`_ for more info.
.. _NOSTRADAMUS: http://ladisk.si/?what=incfl&flnm=nostradamus.php

Created on Wed 19 07 2022

@author: TOMAC Ivan, SLAVIČ Janko
..meta::
    :keywords: Morlet-wave, modal identification, modal parameters, over-determination, noise
"""
import numpy as np
import mwdi as mw

from scipy.special import erf
from scipy.optimize import least_squares
from warnings import warn

class MorletWaveModal(object):

    def __init__(self, free_response, fs, n_1=5, n_2=10, k_lo=10, k_hi=400, num_k=10):
        """
        Initiates the MorletWaveModal object

        :param free_response: analyzed signal
        :param fs:  frequency of sampling
        :param n_1: time spread parameter 1
        :param n_2: time spread parameter 2
        :param k_lo: starting `k` value
        :param k_hi: ending `k` value
        :param num_k: number of distributed `k` values in `[k_lo, k_hi]` range
        :return:
        """
        self.damp_lim = (0.0002, 0.02)
        self.free_response = free_response
        self.fs = fs
        self.n_1 = n_1
        self.n_2 = n_2
        self.k_lo = k_lo
        self.k_hi = k_hi
        self.num_k = num_k
        self.omega_id = None
        
        self.identifier_morlet_wave = mw.MorletWave(free_response, fs)
        return None

    def identify_modal_parameters(self, omega_estimated, damping_estimated=0.0025):
        """
        Wrapper method which performs identification of modal parameters for selected mode.

        :param omega_estimated: initial natural circular frequency
        :param damping_estimated: Damping ratio estimated for selected mode
        :return omega_identified, delta_identified, amplitude_identified, phase_identified: 
        """
        run_steps = True
        while run_steps:
            ### Step #1 ###
            self.initialization(omega_estimated, damping_estimated)

            ### Step #2 ###
            omega_identified = self.identify_natural_frequency(omega_estimated)

            ### Step #3 ###
            damping_identified = self.identify_damping()

            similar = np.isclose(damping_estimated, damping_identified, \
                rtol=0.1, atol=0)
            if damping_estimated < damping_identified and not similar:
                warn(f'Estimated damping: {damping_estimated:.5} is smaller then ' \
                     f'identified: {damping_identified:.5}. Returning to step #1.')
                damping_estimated = damping_identified
                omega_estimated = omega_identified
            else:
                run_steps = False

        if damping_estimated > damping_identified and not similar:
            warn(f'Estimated damping: {damping_estimated:.5} is higher then ' \
                 f'identified: {damping_identified:.5}, possible higher k_lim available.')

        ### Step #4 ###
        amplitude_identified, phase_identified = self.identify_amplitude_phase(damping_identified)

        return omega_identified, damping_identified, amplitude_identified, phase_identified

    def initialization(self, omega_estimated, damping_estimated):
        """
        Step #1: Selection of the estimated damping ratio, determination of `k_lim` and distribution of `k_j`

        :param omega_estimated: initial natural circular frequency
        :param damping_estimated: Damping ratio estimated for selected mode
        :return:
        """
        self.k_hi = None # legacy: prevents k distribution in identify_natural_frequency()

        if self.damp_lim[0] <= damping_estimated < 0.0025:
            k_lim = 400
        elif 0.0025 <= damping_estimated <= self.damp_lim[1]:
            k_lim = int(k_limit_theoretical(self.n_1, damping_estimated) * 1)
        elif self.damp_lim[0] > damping_estimated:
            warn(f'Estimated damping {damping_estimated:.4f} is lower then limit {self.damp_lim[0]:.4f}, using limit.')
            k_lim = 400
        elif damping_estimated > self.damp_lim[1]:
            warn(f'Estimated damping {damping_estimated:.4f} is higher then limit {self.damp_lim[1]:.4f}, using limit.')
            k_lim = int(k_limit_theoretical(self.n_1, self.damp_lim[1]))
        print('k_lim =', k_lim)

        # test k_lim against signal length for the selected mode
        n_signal = self.free_response.size
        k_signal = get_k(omega_estimated, self.fs, n_signal, self.n_1)
        print('k_signal =', k_signal)
        if k_lim > k_signal:
            warn(f'k_lim: {k_lim} exceeds signal length k_signal: {k_signal}. k_lim is adjusted to signal length.')
            k_lim = k_signal
        print('k_lim =', k_lim)

        # check k_lo-k_hi range to avoid double k_j values.
        num_k = k_lim - self.k_lo + 1
        if num_k < self.num_k:
            if num_k < 3:
                raise Exception('Extend k_lo-k_hi range.')
            else:
                raise Exception(f'num_k is too large. Extend k_lo-k_hi range or set k_num to {num_k}.')

        self.k = np.linspace(self.k_lo, k_lim, self.num_k, dtype=int)
        return None

    def identify_natural_frequency(self, omega_estimated, test=False):
        """
        Searches for the natural frequencies in `k` values from range.

        This method distributes `k` values in the selected range. The signal
        length is check against `k_hi` value. Morlet wave integral is 
        determined in all `k` values from the range. The search for maximum
        values of the Morlet-wave integrals for `omega` in `k` range is 
        initiated.
        Natural frequency is determined by averaging frequencies obtained in
        `k` range.

        :param omega_estimated: initial natural circular frequency
        :param test: parameter passed to `morlet_wave_integrate()` method
        :return: identified natural frequency
        """

        if self.k_hi is not None:
            N_ef = self.free_response.size
            k_max = int(omega_estimated * N_ef / (2*np.pi*self.fs))
            k_hi = self.k_hi

            if k_hi > k_max:
                # raise Exception(f'k_hi exceeds the signal length! Reduce k_hi to {int(k_max * 0.9)}.')
                k_hi = int(k_max * 0.9)
                # self.num_k = 7
            num_k = k_hi - self.k_lo + 1
            if num_k < self.num_k:
                if num_k < 3:
                    raise Exception('num_k is too large. Extend k_lo-k_hi range.')
                else:
                    raise Exception(f'num_k is too large. Extend k_lo-k_hi range or set k_num to {num_k}.')
                # self.num_k = self.k_hi - self.k_lo + 1

            self.k = np.linspace(self.k_lo, k_hi, self.num_k, dtype=int)
            # self.k_hi_used = k_hi

        self.find_natural_frequencies(omega_estimated)
        self.morlet_wave_integrate()

        omega_identified = np.average(self.omega_id, weights=self.k)
        return omega_identified

    def identify_damping(self):
        """
        Identifies damping using least square minimization of Eq.(15)

        :return: identified damping
        """
        if self.omega_id is None:
            raise Exception(f'Natural frequencies not identified.')

        damp = least_squares(self.identifier_morlet_wave.exact_mwdi_goal_function, \
            x0=0.001, method='lm', \
            args=(self.integral_ratio, self.n_1, self.n_2, self.k))

        if not(damp.success):
            raise Exception(f'Optimizer returned false:\n{damp.message}.')
        return damp.x[0]

    def identify_amplitude_phase(self, damping):
        """
        Identifies amplitude and phase using LS minimization of Eqs.(16), (17)

        :param damping: identified damping
        :return: identified amplitude and phase
        """

        ######### Amplitude #########
        amp_test = np.mean(get_amplitude(self.k, self.n_1, damping, \
                    self.omega_id, self.integral[0,]))
        amplitude = least_squares(cost_fun_amplitude, x0=amp_test, method='lm', \
                        args=(self.omega_id, damping, self.k, \
                                self.n_1, np.abs(self.integral[0,])))
        if not(amplitude.success):
            raise Exception(f'Optimizer returned false for amplitude:\n{amplitude.message}.')

        ########## Phase ############
        phi_tilde = -np.angle((-1)**(self.k) * self.integral[0,])
        phi_test = np.mean(phi_tilde)
        phase = least_squares(cost_fun_phase, x0=phi_test, method='trf', bounds=(-np.pi, np.pi), \
                    args=(self.k, np.abs(np.tan(phi_test)), phi_tilde))
        if not(phase.success):
            raise Exception(f'Optimizer returned false for phase:\nAmplitude: {amplitude.x[0]}\n{phase.message}.')

        return amplitude.x[0], phase.x[0]

    def find_natural_frequencies(self, omega):
        """
        Searches for natural frequencies around initially defined `omega`.

        :param omega: guessed circular natural frequency
        :return:
        """
        self.omega_id = np.zeros_like(self.k, dtype=float)        
        for i, k_ in enumerate(self.k):
            self.omega_id[i] = self.identifier_morlet_wave.find_natural_frequency(omega, self.n_1, k_)
        return None

    def morlet_wave_integrate(self):
        """
        Calculates the signal based morlet integral ratio, Eq.(10).

        :param test: experimental purpose
        :return:
        """
        if self.omega_id is None:
            raise Exception(f'Natural frequencies not identified.')

        N_hi = get_number_of_samples(self.k[-1], np.min(self.omega_id), self.fs)
        N_k = self.k.size
        psi = np.zeros((N_hi, N_k), dtype=np.complex128)
        self.integral = np.zeros((2, N_k), dtype=np.complex128)
        for j, n_ in enumerate((self.n_1, self.n_2)):
            for i, k_ in enumerate(self.k):
                psi_N = get_number_of_samples(k_, self.omega_id[i], self.fs)
                psi[:psi_N, i] = self.identifier_morlet_wave.morlet_wave(self.omega_id[i], n_, k_)

            temp = np.einsum('i,ij->ij', self.free_response[:N_hi], np.conj(psi))
            self.integral[j,] = np.trapz(temp, dx=1/self.fs, axis=0)
                    
        self.integral_ratio = np.abs(self.integral[0]) / np.abs(self.integral[1])

        return None

def damping_limit_theoretical(n, k):
    """
    Determines the max theoretical damping ration for given `n` and `k`, Eq.(8).

    :param n: time spread parameter
    :param k: number of oscillations
    :return: damping ratio
    """
    kapa = 8*np.pi*k / n**2
    d1 = 0.5*(-kapa + np.sqrt(kapa**2 + 4))
    d2 = 0.5*(-kapa - np.sqrt(kapa**2 + 4))
    if d1 > 0:
        return d1
    elif d2 > 0:
        return d2
    return d1, d2

def k_limit_theoretical(n, d):
    """
    Determines max theoretical oscillations for given `n` and `d`, Eq.(8).

    :param n: time spread parameter
    :param d: damping ratio
    :return: number of oscillations
    """
    kapa = n**2 / (8*np.pi*d)
    return int(kapa * np.sqrt(1 - d**2))

def get_number_of_samples(k, omega, fs):
    """
    Determines number of samples based on `k`, `omega` and `fs`.

    :param k: number of oscillations
    :param omega: circular frequency of signal
    :param fs: sampling frequency of signal in S/s.
    :return: number of samples (S)
    """
    return int(2 * k * np.pi / omega * fs) + 1

def get_k(omega, fs, n_samples, n=0):
    """
    Determines number of oscillation for given parameters.

    Time spread parameter `n` is added to account for frequency spread
    when the optimizer seeks the natural frequency, see Eq.(14), because
    if k_signal = k_lim will result in longer MW function then the signal.

    :param omega: circular frequency of signal
    :param fs: sampling frequency of signal in S/s.
    :param n_samples: number of samples (S)
    :param n: time spread parameter
    :return: number of oscillations
    """
    k = int(omega * n_samples / (2 * np.pi * fs))
    if n==0:
        return k
    else:
        f_spread = frequency_spread_mw(omega, n, k)
    return int((omega - f_spread) * n_samples / (2 * np.pi * fs))

def frequency_spread_mw(omega, n, k):
    """
    Frequency spread of the Morlet-wave function, Eq.(12).

    :param omega: circular frequency
    :param n: time spread parameter
    :param k: number of oscillations
    :return: frequency spread in rad/s
    """
    return 0.25 * omega * n / (np.pi * k)

def get_amplitude(k, n, damping_ratio, omega, I_tilde):
    """
    Determines amplitude from the Morlet-integral.

    :param k: number of oscillations
    :param n: time spread parameter
    :param damping_ratio: damping ratio
    :param omega: circular natural frequency
    :param I_tilde: Morlet integral
    :return: amplitude constant
    """
    const = (0.5 * np.pi)**0.75 * np.sqrt(k / (n * omega))
    e_1 = 2*k*np.pi*damping_ratio / (n*np.sqrt(1 - damping_ratio**2))
    e_2 = 0.25 * n
    err = erf(e_1 + e_2) - erf(e_1 - e_2)
    I_abs = np.exp(e_1**2 - 0.5*n*e_1)
    return np.abs(I_tilde) / (const * I_abs * err)

def cost_fun_amplitude(amplitude, omega, damping_ratio, k, n, I_tilde_abs):
    """
    Cost function for LS minimization of `amplitude`, Eq.(16).

    :param amplitude: amplitude constant of the observed mode
    :param omega: circular natural frequency
    :param damping_ratio: damping ratio
    :param k: number of oscillations
    :param n: time spread parameter
    :param I_tilde_abs: Absolute value of the Morlet integral
    :return: difference between theoretical and signal based Morlet integral
             absolute value.
    """
    norm = (0.5 * np.pi)**0.75 * amplitude * np.sqrt(k / (n * omega))
    e_1 = 2 * k * np.pi * damping_ratio / (n * np.sqrt(1 - damping_ratio**2))
    e_2 = 0.25 * n
    err = erf(e_1 + e_2) - erf(e_1 - e_2)
    I_abs = norm * np.exp(e_1 * (e_1 - 2*e_2)) * err
    return I_abs - I_tilde_abs

def cost_fun_phase(phase, k, test, phi_tilde):
    """
    Cost function for LS minimization of `phase`, Eq.(17).

    :param phase: phase of the observed mode
    :param k: number of oscillations
    :param test: it is a `np.abs(np.tan(np.mean(phi_tilde)))`
    :param phi_tilde: signal based phase angle, obtained from argument of Eq.(6)
    :return: difference between theoretical and signal based Morlet integral
             phase angles cosines or sines.
    """
    phi = np.full_like(k, phase, dtype=float)
    # phi_tilde = -np.angle((-1)**(k) * I_tilde)
    # test = np.abs(np.tan(np.mean(phi_tilde)))
    if test > 1:
        return np.cos(phi) - np.cos(phi_tilde)
    return np.sin(phi) - np.sin(phi_tilde)

if __name__ == "__main__":
    fs = 5000
    t_signal = 1
    w_d = 2 * np.pi * 100
    damping_ratio = 0.005
    amplitude = 1
    pha = np.deg2rad(np.random.default_rng().integers(-180, 180, endpoint=True))
    w_n = w_d / np.sqrt(1 - damping_ratio**2)
    t = np.arange(int(t_signal * fs)) / fs
    free_response = np.cos(w_d*t - pha) * np.exp(-damping_ratio*w_n*t) * amplitude

# Initialize MorletWaveModal
    identifier = MorletWaveModal(free_response=free_response, fs=fs)

# Identify modal parameters
    omega, damping, amp, phi = identifier.identify_modal_parameters(omega_estimated=w_d+0.5, damping_estimated=0.01)

# Print results
    print(f'Natural frequency:\n\tTheorethical: {w_d/(2*np.pi)} Hz\n\tIdentified: {omega/(2*np.pi):.2f} Hz')
    print(f'Damping ratio:\n\tTheorethical: {damping_ratio*100}%\n\tIdentified: {damping*100:.4f}%')
    print(f'Amplitude:\n\tTheorethical: {amplitude}\n\tIdentified: {amp:.2f}')
    print(f'Phase:\n\tTheorethical: {np.rad2deg(pha):.2f} deg\n\tIdentified: {np.rad2deg(phi):.2f} deg')