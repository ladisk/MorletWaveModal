#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import mwdi as mw

from scipy.special import erf

# def damping_limit_theoretical(n, k):
#     """
#     Determines the max theoretical damping ration for given `n` and `k`, Eq.(8).

#     :param n: time spread parameter
#     :param k: number of oscillations
#     :return: damping ratio
#     """
#     kapa = 8*np.pi*k / n**2
#     d1 = 0.5*(-kapa + np.sqrt(kapa**2 + 4))
#     d2 = 0.5*(-kapa - np.sqrt(kapa**2 + 4))
#     if d1 > 0:
#         return d1
#     elif d2 > 0:
#         return d2
#     return d1, d2

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
        f_spread = mw.frequency_spread_mw(omega, n, k)
    return int((omega - f_spread) * n_samples / (2 * np.pi * fs)) - 1

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
