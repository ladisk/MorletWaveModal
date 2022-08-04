MWModal - Morlet-Wave Modal Identification 
------------------------------------------
This is the Python implementation of the Morlet-Wave Modal identification method which is based on the [1]_.

This package is created within the MSCA IF project `NOSTRADAMUS`_.


Simple example
---------------
A simple example how to identify modal parameters using Morlet-Wave Modal package:

.. code-block:: python

    import mwmodal as mwm
    import numpy as np

    # set time domain
    fs = 5000 # sampling frequency [Hz]
    N = 5000 # number of data points of time signal
    time = np.arange(N) / fs # time vector

    # generate a free response of a SDOF damped mechanical system
    w_n = 2*np.pi * 100 # undamped natural frequency
    d = 0.01 # damping ratio
    x = 1 # amplitude
    phi = 0.3 # phase
    response = x * np.exp(-d * w_n * time) * np.cos(w_n * np.sqrt(1 - d**2) * time - phi)

    # set MorletWaveModal object identifier
    identifier = mwm.MorletWaveModal(free_response=response, fs=fs)

    # Step one: Initialize identification
    identifier.initialize_identification(w_n, damping_estimated=0.005)

    # Step two: Identify natural frequency
    omega = identifier.identify_natural_frequency(w_d)
    print(omega)

    # Step three: Identify damping
    damp = identifier.identify_damping()
    print(damp)

    # Step four: Identify amplitude and phase
    amp, pha = identifier.identify_amplitude_phase(damp)
    print(amp, pha)

References
----------
.. [1] J\. Slavič, M. Boltežar, Damping identification with the Morlet-wave, Mechanical Systems and Signal Processing, 25 (2011) 1632–1645, doi: `10.1016/j.ymssp.2011.01.008`_.

.. _NOSTRADAMUS: http://ladisk.si/?what=incfl&flnm=nostradamus.php
.. _10.1016/j.ymssp.2011.01.008: https://doi.org/10.1016/j.ymssp.2011.01.008
   

