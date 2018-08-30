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

class FHDDM(SuperDetector):
    """The Fast Hoeffding Drift Detection Method (FHDDM) class."""

    DETECTOR_NAME = 'FHDDM'

    def __init__(self, window_size=100, delta=0.000001):

        super().__init__()

        self.__DELTA = delta
        self.__WINDOW_SIZE = window_size
        self.__E = math.sqrt(math.log((1 / self.__DELTA), math.e) / (2 * self.__WINDOW_SIZE))

        self.__WINDOW = FastBuffer(max_size=self.__WINDOW_SIZE)
        self.__MU_M = 0

    def run(self, pr):

        drift_status = False

        self.__WINDOW.add_element(pr)

        if self.__WINDOW.isfull():
            mu_t = self.__WINDOW.get_queue().count(True) / self.__WINDOW_SIZE
            if self.__MU_M < mu_t:
                self.__MU_M = mu_t
            drift_status = (self.__MU_M - mu_t) > self.__E

        return drift_status

    def reset(self):
        super().reset()
        self.__WINDOW.clear_queue()
        self.__MU_M = 0

    def get_settings(self):
        settings = [str(self.__WINDOW_SIZE) + "." + str(self.__DELTA),
                    "$n$:" + str(self.__WINDOW_SIZE) + ", " +
                    "$\delta$:" + str(self.__DELTA).upper()]
        return settings
