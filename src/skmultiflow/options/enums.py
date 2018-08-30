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
    # OZA_BOOSTING=auto() # TODO: uncomment when implemented
    GAUSSIAN_NB=auto()
    SGD=auto()
    Multinomial_NB=auto()
    OZA_BAGGING=auto()
    LEVERAGE_BAGGING=auto()
    VOTING_ENSEMBLE=auto()