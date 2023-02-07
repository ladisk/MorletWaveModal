import sys, os

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import numpy as np
import mwmodal

def test_version():
    """ check if MWDI exposes a version attribute """
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

# def test_sythetic(fs=5000, n=5000, fr=100, damping_ratio=0.01, phase=0.3, amplitude=1):
#     w_d = 2*np.pi*fr
#     w_n = w_d / np.sqrt(1 - damping_ratio**2)
#     signal = get_free_response(fs=fs, n=n, w_n=w_n, damping_ratio=damping_ratio, 
#                                phase=phase, amplitude=amplitude)

#     identifier = mwmodal.MorletWaveModal(free_response=signal, fs=fs)
#     # mp = identifier.identify_modal_parameters(omega_estimated=w_n, damping_estimated=damping_ratio)

#     identifier.initialization(omega_estimated=w_d+1, damping_estimated=damping_ratio)
#     np.testing.assert_equal(identifier.k[-1], 98)
#     print('Initialization test: PASSED')

#     omega_identified = identifier.identify_natural_frequency(w_d+1)
#     np.testing.assert_allclose(omega_identified, w_d, 1.7e-4)
#     print(f'\nNatural frequency identification test: PASSED\n\tomega={w_d:.2f}, omega_identified={omega_identified:.2f}')

#     damping_ratio_identified = identifier.identify_damping()
#     np.testing.assert_allclose(damping_ratio_identified, damping_ratio, 1.5e-3)
#     print(f'\nDamping identification test: PASSED\n\tdelta={damping_ratio*100}%, delta_identified={damping_ratio_identified*100:.4f}')

#     amplitude_identified, phase_identified = identifier.identify_amplitude_phase(damping_ratio_identified)
#     np.testing.assert_allclose(amplitude_identified, amplitude, 2.4e-3)
#     print(f'\nAmplitude identification test: PASSED\n\tX={amplitude}, X_identified={amplitude_identified:.4f}')

#     np.testing.assert_allclose(phase_identified, phase, 6.8e-2)
#     print(f'\nPhase identification test: PASSED\n\tphase={phase:.2f}, phase_identified={phase_identified:.2f}')

def test_multi_sine(fs=40000, n=50000):
    wd = 2*np.pi*np.array([100, 300, 600, 800])
    damping_ratio = [0.01, 0.015, 0.02, 0.005]
    phase = [0.3, 1.0, 1.4, 2.2]
    amplitude = [1, .5, 0.7, 0.2]

    signal = np.zeros(n)
    for wd_, d_, p_, a_ in zip(wd, damping_ratio, phase, amplitude):
        wn = wd_ / np.sqrt(1 - d_**2)
        signal += get_free_response(fs=fs, n=n, w_n=wn, damping_ratio=d_, phase=p_, amplitude=a_)

    identifier = mwmodal.MorletWaveModal(free_response=signal, fs=fs)
    k_lim = [99, 66, 49, 198]
    for wd_, d_, p_, a_, k_ in zip(wd, damping_ratio, phase, amplitude, k_lim):
        print('')
        identifier.initialization(omega_estimated=wd_+1, damping_estimated=d_)
        np.testing.assert_equal(identifier.k[-1], k_)
        print('Initialization test: PASSED')

        omega_identified = identifier.identify_natural_frequency(wd_+1)
        np.testing.assert_allclose(omega_identified, wd_, 6.5e-4)
        print(f'Natural frequency identification test:\t PASSED\t(omega={wd_:.2f}, omega_identified={omega_identified:.2f})')

        identifier.morlet_wave_integrate()
        print('Integration: PASSED')

        damping_ratio_identified = identifier.identify_damping()
        np.testing.assert_allclose(damping_ratio_identified, d_, 5.8e-2)
        print(f'Damping identification test:\t\t PASSED\t(delta={d_*100}%, delta_identified={damping_ratio_identified*100:.4f}%)')

        amplitude_identified, phase_identified = identifier.identify_amplitude_phase(damping_ratio_identified)
        np.testing.assert_allclose(amplitude_identified, a_, 1.1e-1)
        print(f'Amplitude identification test:\t\t PASSED\t(amplitude={a_}, amplitude_identified={amplitude_identified:.4f})')

        np.testing.assert_allclose(phase_identified, p_, 6.8e-2)
        print(f'Phase identification test:\t\t PASSED\t(phase={p_}, phase_identified={phase_identified:.2f})')


if __name__ == "__main__":
    #test_version()
    # test_sythetic()
    test_multi_sine()

if __name__ == '__mains__':
    np.testing.run_module_suite()
