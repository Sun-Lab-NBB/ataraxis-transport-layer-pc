"""Playground to test some low level developments before adding them to the main files"""

import sys
from threading import Event, Timer
from helper_modules import ElapsedTimer
import time as tm


def my_blocker():
    event = Event()
    for _ in range(50):
        start = tm.perf_counter_ns()
        event.wait(timeout=1e-6)
        end = tm.perf_counter_ns()
        print((end-start)/1000)


my_blocker()