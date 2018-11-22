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
    PARTIAL=auto()
    ALL=auto()
    BLIND_RANDOM=auto()
    NONE=auto()

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
    MULTINOMIAL_NB=auto()
    GAUSSIAN_NB=auto()
    SGD=auto()
    LEVERAGE_BAGGING=auto()
    NO_CHANGE=auto()
    MAJORITY_VOTING=auto()