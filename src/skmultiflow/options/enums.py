from enum import Enum, auto

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

class Voting(Enum):
    """
    Voting classifier voting technique
    """
    AVG_W_PROBABILITY=auto()
    W_AVG_PROBABILITY=auto()
    BOOLEAN=auto()
    PROBABILITIY=auto()

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

class DriftContent(Enum):
    """
    Drift detection content
    """
    BOOLEAN=auto()
    PROBABILITY=auto()
    WEIGHTED_PROBABILITY=auto()

class DriftDetectorCount(Enum):
    """
    Number of drift detectors in the voting classifier
    """
    ONE_PER_CLASSIFIER=auto()
    ONE_FOR_ENSEMBLE=auto()