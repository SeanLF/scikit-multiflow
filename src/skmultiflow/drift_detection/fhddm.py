"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Fast Hoeffding Drift Detection Method (FHDDM) Implementation ***
Paper: Pesaranghader, Ali, and Herna L. Viktor. "Fast hoeffding drift detection method for evolving data streams."
Published in: Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer International Publishing, 2016.
URL: https://link.springer.com/chapter/10.1007/978-3-319-46227-1_7
"""

import math
from skmultiflow.drift_detection.detector import SuperDetector
from skmultiflow.utils.data_structures import FastBuffer
import numpy as np

class FHDDM(SuperDetector):
    """The Fast Hoeffding Drift Detection Method (FHDDM) class."""

    DETECTOR_NAME = 'FHDDM'

    def __init__(self, window_size=100, delta=0.000001):

        super().__init__()

        self.DELTA = delta
        self.WINDOW_SIZE = window_size
        self.E = math.sqrt(math.log((1 / self.DELTA), math.e) / (2 * self.WINDOW_SIZE))

        self.WINDOW = FastBuffer(max_size=self.WINDOW_SIZE)
        self.MU_MAX = 0

    def run(self, pr):

        drift_status = False

        self.WINDOW.add_element(pr)

        if self.WINDOW.is_full():
            mu_current = self.WINDOW.get_queue().count(True) / self.WINDOW_SIZE
            if self.MU_MAX < mu_current:
                self.MU_MAX = mu_current
            drift_status = (self.MU_MAX - mu_current) > self.E

        return drift_status

    def reset(self):
        super().reset()
        self.WINDOW.clear_queue()
        self.MU_MAX = 0

    def get_settings(self):
        settings = [str(self.WINDOW_SIZE) + "." + str(self.DELTA),
                    "$n$:" + str(self.WINDOW_SIZE) + ", " +
                    "$\delta$:" + str(self.DELTA).upper()]
        return settings

class FHDDMS(SuperDetector):
    """The Stacking Fast Hoeffding Drift Detection Method (FHDDMS) class."""

    DETECTOR_NAME = 'FHDDMS'

    def __init__(self, small_window_size=25, large_window_size=100, delta=0.000001):

        super().__init__()

        self.DELTA = delta
        self.SMALL_WINDOW_SIZE = small_window_size
        self.LARGE_WINDOW_SIZE = large_window_size
        self.Es = math.sqrt(math.log((1 / self.DELTA), math.e) / (2 * self.SMALL_WINDOW_SIZE))
        self.El = math.sqrt(math.log((1 / self.DELTA), math.e) / (2 * self.LARGE_WINDOW_SIZE))

        self.WINDOW = FastBuffer(max_size=self.LARGE_WINDOW_SIZE)

        self.SMALL_MU_MAX = 0
        self.LARGE_MU_MAX = 0

    def run(self, pr):

        drift_status = False

        self.WINDOW.add_element(pr)

        if self.WINDOW.is_full():
            small_mu_current = self.WINDOW.get_queue()[(self.LARGE_WINDOW_SIZE-self.SMALL_WINDOW_SIZE):].count(True) / self.SMALL_WINDOW_SIZE
            large_mu_current = self.WINDOW.get_queue().count(True) / self.LARGE_WINDOW_SIZE
            if self.SMALL_MU_MAX < small_mu_current:
                self.SMALL_MU_MAX = small_mu_current
            if self.LARGE_MU_MAX < large_mu_current:
                self.LARGE_MU_MAX = large_mu_current
            drift_status = ((self.SMALL_MU_MAX - small_mu_current) > self.Es) or ((self.LARGE_MU_MAX - large_mu_current) > self.El) 

        return drift_status

    def reset(self):
        super().reset()
        self.WINDOW.clear_queue()
        self.SMALL_MU_MAX = 0
        self.LARGE_MU_MAX = 0

    def get_settings(self):
        settings = [str(self.DELTA),
                    "$ns$:" + str(self.SMALL_WINDOW_SIZE) + ", " +
                    "$nl$:" + str(self.LARGE_WINDOW_SIZE) + ", " +
                    "$\delta$:" + str(self.DELTA).upper()]
        return settings

class PFHDDM(SuperDetector):
    """The Probability Fast Hoeffding Drift Detection Method (PFHDDM) class."""

    DETECTOR_NAME = 'PFHDDM'

    def __init__(self, window_size=100, delta=0.000001):

        super().__init__()

        self.DELTA = delta
        self.WINDOW_SIZE = window_size
        self.E = math.sqrt(math.log((1 / self.DELTA), math.e) / (2 * self.WINDOW_SIZE))

        self.WINDOW = FastBuffer(max_size=self.WINDOW_SIZE)
        self.MU_MAX = 0

    def run(self, pr):

        drift_status = False

        self.WINDOW.add_element(pr)

        if self.WINDOW.is_full():
            mu_current = np.average(self.WINDOW.get_queue())
            if self.MU_MAX < mu_current:
                self.MU_MAX = mu_current
            drift_status = (self.MU_MAX - mu_current) > self.E

        return drift_status

    def reset(self):
        super().reset()
        self.WINDOW.clear_queue()
        self.MU_MAX = 0

    def get_settings(self):
        settings = [str(self.WINDOW_SIZE) + "." + str(self.DELTA),
                    "$n$:" + str(self.WINDOW_SIZE) + ", " +
                    "$\delta$:" + str(self.DELTA).upper()]
        return settings

class PFHDDMS(SuperDetector):
    """The Probability Fast Hoeffding Drift Detection Method (PFHDDMS) class."""

    DETECTOR_NAME = 'PFHDDMS'

    def __init__(self, small_window_size=25, large_window_size=100, delta=0.02):

        super().__init__()

        self.DELTA = delta
        self.SMALL_WINDOW_SIZE = small_window_size
        self.LARGE_WINDOW_SIZE = large_window_size
        self.Es = math.sqrt(math.log((1 / self.DELTA), math.e) / (2 * self.SMALL_WINDOW_SIZE))
        self.El = math.sqrt(math.log((1 / self.DELTA), math.e) / (2 * self.LARGE_WINDOW_SIZE))

        self.WINDOW = FastBuffer(max_size=self.LARGE_WINDOW_SIZE)
        # self.tiny_stats = [[0, '', 0], [0, '', 0]]

        self.SMALL_MU_MAX = 0
        self.LARGE_MU_MAX = 0

    def run(self, pr):

        drift_status = False

        self.WINDOW.add_element(pr)

        if self.WINDOW.is_full():
            small_mu_current = np.average(self.WINDOW.get_queue()[(self.LARGE_WINDOW_SIZE-self.SMALL_WINDOW_SIZE):])
            large_mu_current = np.average(self.WINDOW.get_queue())
            if self.SMALL_MU_MAX < small_mu_current:
                self.SMALL_MU_MAX = small_mu_current
            if self.LARGE_MU_MAX < large_mu_current:
                self.LARGE_MU_MAX = large_mu_current
            small = (self.SMALL_MU_MAX - small_mu_current)
            large = (self.LARGE_MU_MAX - large_mu_current)
            drift_status = (small > self.Es) or (large > self.El)
            # print(self.stats(0, small_mu_current), self.stats(1, large_mu_current))
        return drift_status

    # def stats(self, index, mu):
    #     diff = mu - self.tiny_stats[index][0]
    #     to_print = ''

    #     if diff > 0:
    #         sign = '+'
    #     elif diff < 0:
    #         sign = '-'
    #     else:
    #         sign = ''

    #     if self.tiny_stats[index][1] == sign:
    #         self.tiny_stats[index][2] += 1
    #     else:
    #         to_print = str(self.tiny_stats[index][1]) + str(self.tiny_stats[index][2])
    #         self.tiny_stats[index][2] = 1
    #     self.tiny_stats[index][1] = sign
    #     self.tiny_stats[index][0] = mu
    #     return to_print

    def reset(self):
        super().reset()
        self.WINDOW.clear_queue()
        self.SMALL_MU_MAX = 0
        self.LARGE_MU_MAX = 0

    def get_settings(self):
        settings = [str(self.DELTA),
                    "$ns$:" + str(self.SMALL_WINDOW_SIZE) + ", " +
                    "$nl$:" + str(self.LARGE_WINDOW_SIZE) + ", " +
                    "$\delta$:" + str(self.DELTA).upper()]
        return settings