import numpy as np
import pytest
import mwmodal


def test_version():
    """Check that mwmodal exposes a version string."""
    assert hasattr(mwmodal, '__version__')
    assert isinstance(mwmodal.__version__, str)


def get_free_response(fs=5000, n=5000, w_n=100, damping_ratio=0.01, phase=0.3, amplitude=1.0):
    """
    Calculates a time response of free SDOF damped mechanical system.

    :param fs: sample frequency (S/s)
    :param n: number of samples that will response contain (S)
    :param w_n: undamped natural circular frequency (Hz)
    :param damping_ratio: damping ratio of the system (-)
    :param phase: phase of the response (rad)
    :param amplitude: amplitude of the response
    """
    time = np.arange(n) / fs
    w_d = w_n * np.sqrt(1 - damping_ratio**2)
    free_response = amplitude * np.cos(w_d * time - phase) * np.exp(-damping_ratio * w_n * time)
    return free_response


def test_multi_sine():
    fs = 40000
    n = 50000
    wd = 2 * np.pi * np.array([100, 300, 600, 800])
    damping_ratio = [0.01, 0.015, 0.02, 0.005]
    phase = [0.3, 1.0, 1.4, 2.2]
    amplitude = [1, 0.5, 0.7, 0.2]

    signal = np.zeros(n)
    for wd_, d_, p_, a_ in zip(wd, damping_ratio, phase, amplitude):
        wn = wd_ / np.sqrt(1 - d_**2)
        signal += get_free_response(fs=fs, n=n, w_n=wn, damping_ratio=d_, phase=p_, amplitude=a_)

    identifier = mwmodal.MorletWaveModal(free_response=signal, fs=fs)
    k_lim = [99, 66, 49, 198]
    for wd_, d_, p_, a_, k_ in zip(wd, damping_ratio, phase, amplitude, k_lim):
        identifier.initialization(omega_estimated=wd_ + 1, damping_estimated=d_)
        np.testing.assert_equal(identifier.k[-1], k_)

        omega_identified = identifier.identify_natural_frequency(wd_ + 1)
        np.testing.assert_allclose(omega_identified, wd_, rtol=6.5e-4)

        identifier.morlet_wave_integrate()

        damping_ratio_identified = identifier.identify_damping()
        np.testing.assert_allclose(damping_ratio_identified, d_, rtol=5.8e-2)

        amplitude_identified, phase_identified = identifier.identify_amplitude_phase(damping_ratio_identified)
        np.testing.assert_allclose(amplitude_identified, a_, rtol=1.1e-1)
        np.testing.assert_allclose(phase_identified, p_, rtol=6.8e-2)

if __name__ == '__mains__':
    np.testing.run_module_suite()
