from enum import Enum, auto

class Window(Enum):
    """
    Streaming window type
    """
    SLIDING=auto()
    TUMBLING=auto()
    HYBRID=auto()

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
    BEFORE_WEIGHT='before_weight'
    AFTER_WEIGHT='after_weight'
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
    NO_CHANGE=auto()
    MAJORITY_VOTING=auto()
    HOEFFDING_TREE=auto()