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

from scipy.optimize import least_squares
from warnings import warn

from .tools import *

class MorletWaveModal(object):

    def __init__(self, free_response, fs, n_1=5, n_2=10, k_lo=10, num_k=10):
        """
        Initiates the MorletWaveModal object

        :param free_response: analyzed signal
        :param fs:  frequency of sampling
        :param n_1: time spread parameter 1
        :param n_2: time spread parameter 2
        :param k_lo: starting `k` value
        :param num_k: number of distributed `k` values in `[k_lo, k_hi]` range
        :return:
        """
        self.damp_lim = (0.0002, 0.02)
        self.free_response = free_response
        self.fs = fs
        self.n_1 = n_1
        self.n_2 = n_2
        self.k_lo = k_lo
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
            self.morlet_wave_integrate()
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
        Step #1: Selection of the estimated damping ratio, determination of `k_lim` and 
        distribution of `k_j` values in range `[k_lo, k_hi=k_lim]`

        :param omega_estimated: initial natural circular frequency
        :param damping_estimated: Damping ratio estimated for selected mode
        :return:
        """

        if self.damp_lim[0] <= damping_estimated < 0.0025:
            k_lim = 400
        elif 0.0025 <= damping_estimated <= self.damp_lim[1]:
            k_lim = k_limit_theoretical(self.n_1, damping_estimated) * 1
        elif self.damp_lim[0] > damping_estimated:
            warn(f'Estimated damping {damping_estimated:.4f} is lower then limit {self.damp_lim[0]:.4f}, using limit.')
            k_lim = 400
        elif damping_estimated > self.damp_lim[1]:
            warn(f'Estimated damping {damping_estimated:.4f} is higher then limit {self.damp_lim[1]:.4f}, using limit.')
            k_lim = k_limit_theoretical(self.n_1, self.damp_lim[1])
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

    def identify_natural_frequency(self, omega_estimated):
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
        :return: identified natural frequency
        """

        self.find_natural_frequencies(omega_estimated)
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
        if self.omega_id is None:
            raise Exception(f'Natural frequencies not identified.')

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
        N_response = self.free_response.size
        if N_hi > N_response:
            raise Exception(f'Wave function is larger {N_hi} then signal {N_response}.\n' \
                f'Omega: {np.around(self.omega_id, 1)}.\nPossible k_lo={self.k_lo} is too low, try increase.')
        
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
