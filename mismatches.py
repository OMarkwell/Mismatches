import numpy as np
import scipy.optimize
import scipy.interpolate as interp
from functools import partial


def overlap(h1, h2, dt):
    return np.trapezoid(h1 *np.conjugate(h2), dx = dt)
    

def compute_mismatch(h1, h2, dt):
    n1 = overlap(h1, h1, dt)
    n2 = overlap(h2, h2, dt)
    h1h2 = overlap(h1, h2, dt)
    return (1 - np.real(h1h2 / np.sqrt(n1*n2)))


def interp_waveform(h, t_wf, t_target):

    h_interp = interp.splrep(t_wf, h)
    return interp.splev(t_target, h_interp)



def shift_and_mismatch(h1, h2, t1, t2, t_start, t_end, shift):
""" Time and phase shifts the waveform h2 and then computes the TD mismatch over (t_start, t_end)
Parameters
------------
h1, h2 - Arrays representing the waveforms to be compared
t1, t2 - Arrays representing the time on which h1 and h2 are evaluated
t_start, t_end - Floats, lower and upper bounds of mismatch integral
shift - Double [dt, dphi], the desired time and phase translation

Returns
-----------
mm - float, The mismatch between the two waveforms
"""
    h1 = h1.copy()
    h2 = h2.copy()
    t1 = t1.copy()
    t2 = t2.copy()

    dt = min(np.min(np.diff(t1)), np.min(np.diff(t2)))

    delta_t = shift[0]
    delta_phi = shift[1]

    t2 -= delta_t
    h2 *= np.exp(1.j*delta_phi)

    h1 = uniform_grid(h1, t1, t_start, t_end, dt)
    h2 = uniform_grid(h2, t2, t_start, t_end, dt)

    mm = compute_mismatch(h1, h2, dt)

    return mm

def uniform_grid(h,t, t_start, t_end, dt):
    t_new = np.arange(t_start, t_end, dt)
    return interp_waveform(h, t, t_new)

def optimized_mismatch(h1, h2, t1, t2, t_start = None, t_end = None):
""" Computes the TD mismatch between h1 and h2 optimized over time and phase
Parameters
------------
h1, h2 - Arrays representing the waveforms to be compared
t1, t2 - Arrays representing the time on which h1 and h2 are evaluated
t_start, t_end - Floats, lower and upper bounds of mismatch integral

Returns
------------
min_mm - Float, the minimized mismatch

"""
    if not t_start:
        t_start = max(t1[0], t2[0]) + 100
    if not t_start:
        t_end = min(t1[-1], t2[-1]) - 100

    ##This is probably excessive but it's all fairly quick.
    dphi_guesses = np.linspace(0, 2*np.pi, 4, endpoint = False)
    dt_guesses = np.linspace(-25, 25, 3)


    min_dx = t_end - t2[-1]
    max_dx = t_start - t2[0]
    min_mm = 100

    for dt in dt_guesses:
        for dphi in dphi_guesses:
            guess = [dt, dphi]
            x = scipy.optimize.minimize(partial(shift_and_mismatch,h1, h2, t1, t2, t_start, t_end), guess, bounds = [[min_dx, max_dx],[0, 2*np.pi]] )
            if x.fun < min_mm:
                min_mm = x.fun
    return min_mm



if __name__ == "__main__":

    t1 = np.linspace(-1000, 100, 1234)
    t2 = np.linspace(-1324, 89, 1111)

    h1 = np.cos(t1) + 1j*np.sin(t1)
    h2 = np.exp(1j*np.pi  / 4) *(np.cos(t2 + 5) + 1j*np.sin(t2 + 5))

    t_start = -900
    t_end = 50

    print(optimized_mismatch(h1, h2, t1, t2, t_start = -900, t_end = 50))

