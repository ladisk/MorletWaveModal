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
    T = 1 # signal duration [s]
    time = np.arange(T*fs) / fs # time vector

    # generate a free response of a SDOF damped mechanical system
    w_d = 2*np.pi * 100 # damped natural frequency
    d = 0.01 # damping ratio
    x = 1 # amplitude
    phi = 0.3 # phase
    response = x * np.exp(-d * w_d / np.sqrt(1 - d**2) * time) * np.cos(w_d * time - phi)

    # set MorletWaveModal object identifier
    identifier = mwm.MorletWaveModal(free_response=response, fs=fs)

    #  set initial natural frequency, estimate damping ratio and identify modal parameters
    identifier.identify_modal_parameters(omega_estimated=w_n, damping_estimated=0.01)

References
----------
.. [1] J\. Slavič, M. Boltežar, Damping identification with the Morlet-wave, Mechanical Systems and Signal Processing, 25 (2011) 1632–1645, doi: `10.1016/j.ymssp.2011.01.008`_.


.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7002905.svg
   :target: https://doi.org/10.5281/zenodo.7002905

.. _NOSTRADAMUS: http://ladisk.si/?what=incfl&flnm=nostradamus.php
.. _10.1016/j.ymssp.2011.01.008: https://doi.org/10.1016/j.ymssp.2011.01.008
