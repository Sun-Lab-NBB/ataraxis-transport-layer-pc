"""Playground to test some low level developments before adding them to the main files"""

from threading import Event, Timer

import numpy as np

from helper_modules import ElapsedTimer
import time as tm
from numba import njit

@njit
def long_calculation():
    array = np.zeros(10000, dtype=np.uint32)
    for i in range(10000,):
        array[i] = i


def my_blocker():
    for _ in range(10):
        start = tm.perf_counter_ns()
        long_calculation()
        end = tm.perf_counter_ns()
        print((end-start)/1000)


my_blocker()
