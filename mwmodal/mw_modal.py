#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import mwdi as mw

from scipy.special import erf
from scipy.optimize import least_squares
# from scipy.optimize import minimize_scalar
# from scipy.optimize import minimize

class MorletWaveModal(object):

    def __init__(self, free_response, fs, n_1=5, n_2=10, k_lo=30, k_hi=400, num_k=10):
        """
        Initiates the MorletWave object

        :param free_response: analysed signal
        :param fs:  frequency of sampling
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
        
        return None

    def identify_natural_frequency(self, omega):
        
        N_ef = free_response.size
        k_max = int(omega * N_ef / (2*np.pi*fs))

        if self.k_hi > k_max:
            self.k_hi = int(k_max * 0.9)
            self.num_k = 7
        if k_hi - self.k_lo + 1 < self.num_k:
            self.num_k = k_hi - self.k_lo + 1

        self.k = np.linspace(self.k_lo, self.k_hi, self.num_k, dtype=int)

        self.find_natural_frequencies(omega)
        self.mw_integrate()

        omega_identified = np.mean(self.omega_id[self.mask])
        return omega_identified

    def identify_damping(self):
        if self.omega_id is None:
            raise Exception(f'Natural frequencies not identified.')

        damp = least_squares(cost_fun_damping, x0=0.001, method='lm', \
            args=(self.integral_ratio[self.mask], self.n_1, self.n_2, self.k[self.mask]))

        if not(damp.success):
            raise Exception(f'Optimizer returned false:\n{damp.message}.')
        return damp.x[0]

    def identify_amp_pha(self, damping):

        ######### Amplitude #########
        amp_test = get_amplitude(self.k[self.mask], self.n_1, damping, \
            self.omega_id[self.mask], self.integral[0, self.mask])
        amp = least_squares(cost_fun_amplitude, x0=amp_test, method='lm', \
            args=(self.omega_id[self.mask], damping, self.k[self.mask], self.n_1, \
                np.abs(self.integral[0, self.mask])))

        if not(amp.success):
            raise Exception(f'Optimizer returned false for amplitude:\n{amp.message}.')

        ######### Phase #########
        # def phase_goal_fun(phase, k, I_num):
        #     I_anl_arg = np.exp(1j * (np.pi*k - phase))
        #     return np.sum(np.abs(np.angle(I_anl_arg) - np.angle(I_num)))
        #     # return np.sum(np.square(np.angle(I_anl_arg) - np.angle(I_num)))

        # pha = minimize(fun=phase_goal_fun, x0=0, \
        #     args=(self.k[self.mask], self.integral[0, self.mask]), method='Nelder-Mead')
            
        # if pha.success and -np.pi <= pha.x[0] <= np.pi:
        #     ph = pha.x[0]
        # else:
        #     pha = minimize_scalar(fun=phase_goal_fun, bounds=(-np.pi, np.pi), \
        #             args=(self.k[self.mask], self.integral[0, self.mask]), method='Bounded')
        #     if pha.success:
        #         ph = pha.x
        #     else:
        #         ph = np.nan
        phi_test = 0.1
        phase = least_squares(cost_fun_phase, x0=phi_test, method='trf', bounds=(-np.pi, np.pi),  \
                    args=(self.k[self.mask], np.angle(identifier.integral[0, self.mask])))
        if phase.cost > 1:
            phase = least_squares(cost_fun_phase, x0=-phi_test, method='trf', bounds=(-np.pi, np.pi),  \
                    args=(self.k[self.mask], np.angle(identifier.integral[0, self.mask])))
        if not(phase.success):
            raise Exception(f'Optimizer returned false for phase:\nAmplitude: {amp.x[0]}\n{phase.message}.')

        return amp.x[0], phase.x[0]

    def find_natural_frequencies(self, omega):
        identifier = mw.MorletWave(self.free_response, self.fs)
        self.omega_id = np.zeros_like(self.k, dtype=float)        
        for i, k_ in enumerate(self.k):
            self.omega_id[i] = identifier.find_natural_frequency(omega, self.n_1, k_)
        return None

    def mw_integrate(self):
        if self.omega_id is None:
            raise Exception(f'Natural frequencies not identified.')

        N_hi = max_N(self.k[-1], np.min(self.omega_id), self.fs)
        N_k = self.k.size
        psi = np.zeros((N_hi, N_k), dtype=np.complex128)
        self.integral = np.zeros((2, N_k), dtype=np.complex128)
        for j, n_ in enumerate((self.n_1, self.n_2)):
            for i, k_ in enumerate(self.k):
                psi_N = max_N(k_, self.omega_id[i], self.fs)
                psi[:psi_N, i] = morlet_wave(self.fs, self.omega_id[i], k_, n_)

            self.integral[j,] = \
                np.trapz(np.einsum('i,ij->ij', self.free_response[:N_hi], np.conj(psi)), \
                    dx=1/self.fs, axis=0)
                    
        self.integral_ratio = np.abs(self.integral[0]) / np.abs(self.integral[1])
        self.mask = self.analyze_ratio(self.integral_ratio)

        if np.sum(self.mask) < 3:
            raise Exception(f'Number of points is lower then 3, {np.sum(self.mask)}.')
        return None

    def analyze_ratio(self, M_num):
        k_lim = int(k_limit_theoret(self.n_1, self.damp_lim[1]))
        for i, k_ in enumerate(self.k):
            if k_ <= k_lim:
                test = cost_fun_damping(self.damp_lim[1], 0, self.n_1, self.n_2, k_)
                if M_num[i] > test:
                    M_num[i] = 1
        d_lim = damping_limit_theoret(self.n_1, self.k_hi)
        M_lim_lo = low_M_limit((self.n_1, self.n_2))
        M_lim_hi = cost_fun_damping(d_lim, 0, self.n_1, self.n_2, self.k_hi)
        mask = np.logical_and(M_num > M_lim_lo, M_num < M_lim_hi)
        return mask

def morlet_wave(fs, omega, k, n):
    """
    Function generates Morlet-Wave basic wavelet function on the circular 
    frequency `omega` with `k` cycles and `n` time spread.
    
    :param fs: sampling frequency (S/s)
    :param omega: circular frequency (rad/s)
    :param k: number of oscillations
    :param n: time-spread parameter
    :return:
    Source: 200 - Generate Morlet-Wave numerically - FUNCTION.ipynb
    Date: petek, 18. marec 2022 13:52:07
    """
    eta = 2 * k * np.pi * np.sqrt(2) / n
    s = eta / omega
    T = 2 * k * np.pi / omega
    N = int(fs * T) + 1
    
    t = np.arange(N) / fs
    t -= 0.5 * T
    t /= s
    
    return np.pi**-0.25 * s**-0.5 * np.exp(-0.5*t**2 + 1j*eta*t)

def damping_limit_theoret(n, k):
    kapa = 8*np.pi*k / n**2
    d1 = 0.5*(-kapa + np.sqrt(kapa**2 + 4))
    d2 = 0.5*(-kapa - np.sqrt(kapa**2 + 4))
    if d1 > 0:
        return d1
    elif d2 > 0:
        return d2
    return d1, d2

def k_limit_theoret(n, d):
    kapa = n**2 / (8*np.pi*d)
    return kapa * np.sqrt(1 - d**2)

def low_M_limit(n):
    return erf(0.25*n[0]) / erf(0.25*n[1]) * np.sqrt(n[1] / n[0])

def max_N(k_max, omega, fs):
    return int(2 * k_max * np.pi / omega * fs) + 1

def cost_fun_damping(d, M_num, n_1, n_2, k):  # vektoriziran za eMW2
    const = 2 * k * np.pi * d / np.sqrt(1 - d**2)
    n = np.array([n_1, n_2], dtype=object)
    g_1 = 0.25 * n
    g_2_0 = const / n[0]
    g_2_1 = const / n[1]
    G = erf(g_1[0] - g_2_0) + erf(g_1[0] + g_2_0)
    G /=erf(g_1[1] - g_2_1) + erf(g_1[1] + g_2_1)
    g_2_0 /= n[0]
    g_2_1 /= n[1]
    return np.sqrt(n_2 / n_1) * np.exp(g_2_0 * g_2_1 * (n_2**2 - n_1**2)) * G - M_num

def get_amplitude(k, n, d, w, I):
    const = (np.pi/2)**0.75 * np.sqrt(k / (n * w))
    A = 2*k*np.pi*d / (n*np.sqrt(1 - d**2))
    B = 0.25 * n
    G = erf(A + B) - erf(A - B)
    I_abs = np.exp(A**2 - 0.5*n*A)
    return np.abs(I) / (const * I_abs * G)

def cost_fun_amplitude(amp, omega, damp, k, n, I_tilde_abs):
    norm = (0.5 * np.pi)**0.75 * amp * np.sqrt(k / (n * omega))
    A = 2 * k * np.pi * damp / (n * np.sqrt(1 - damp**2))
    B = 0.25 * n
    G = erf(A + B) - erf(A - B)
    I_anl_abs = norm * np.exp(A * (A - 2*B)) * G
    return I_anl_abs - I_tilde_abs

def cost_fun_phase(phase, k, I_tilde_arg):
    I_anl_arg = np.exp(1j * (np.pi*k - phase))
    return np.angle(I_anl_arg) - I_tilde_arg

if __name__ == "__main__":
    fs = 10000
    k_hi = 100
    w = 2 * np.pi * 100
    damping_ratio = 0.01
    wd = w * np.sqrt(1 - damping_ratio**2)
    amplitude = 1
    pha = np.deg2rad(np.random.randint(-180, 180))
    t = np.arange(0.5*fs) / fs
    free_response = np.cos(wd*t - pha) * np.exp(-damping_ratio*w*t) * amplitude

# Initialize MorletWaveModal
    identifier = MorletWaveModal(free_response=free_response, fs=fs, k_lo=10, k_hi=k_hi)

# Identify damping and natural frequency
    omega = identifier.identify_natural_frequency(omega=w+1)
    damping = identifier.identify_damping()
    print(f'Natural frequency:\n\tTheorethical: {w/(2*np.pi)} Hz\n\tIdentified: {omega/(2*np.pi):.2f} Hz')
    print(f'Damping ratio:\n\tTheorethical: {damping_ratio*100}%\n\tIdentified: {damping*100:.4f}%')

# Identify amplitude and phase
    amp, phi = identifier.identify_amp_pha(damping)
    print(f'Amplitude:\n\tTheorethical: {amplitude}\n\tIdentified: {amp:.2f}')
    print(f'Phase:\n\tTheorethical: {np.rad2deg(pha):.2f} deg\n\tIdentified: {np.rad2deg(phi):.2f} deg')