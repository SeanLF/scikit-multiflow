from enum import Enum, auto

class Window(Enum):
    """
    Streaming window type
    """
    SLIDING=auto()
    TUMBLING=auto()
    SLIDING_TUMBLING=auto()

class DriftReset(Enum):
    """
    Drift detection reset technique
    """
    NONE=auto()
    BLIND=auto()
    PARTIAL=auto()
    ALL=auto()

class Voting(Enum):
    """
    Voting classifier voting technique
    """
    SUM_PROB='sum_prob'
    HARD='hard'
    SOFT='soft'

class Classifier(Enum):
    """
    Classifiers to test in experiments
    """
    VOTING_ENSEMBLE=auto()
    LEVERAGE_BAGGING=auto()
    MULTINOMIAL_NB=auto()
    GAUSSIAN_NB=auto()
    SGD=auto()

    
# OZA_BAGGING=auto()