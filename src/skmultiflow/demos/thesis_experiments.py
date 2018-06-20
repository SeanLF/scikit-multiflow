__author__ = 'Sean Floyd'

from skmultiflow.data.generators.waveform_generator import WaveformGenerator
from skmultiflow.data.generators.sea_generator import SEAGenerator
from skmultiflow.data.file_stream import FileStream
from skmultiflow.demos.thesis import demo

import numpy as np

# Experiments to perform

SLIDING=0
TUMBLING=1
SLIDING_TUMBLING=2
TORNADO_FILES = [
    "/Users/sean/dev/tornado/data_streams/mixed_w_50_n_0.1/mixed_w_50_n_0.1_101.csv",
    "/Users/sean/dev/tornado/data_streams/mixed_w_50_n_0.1/mixed_w_50_n_0.1_103.csv",
    "/Users/sean/dev/tornado/data_streams/mixed_w_50_n_0.1/mixed_w_50_n_0.1_102.csv",
    "/Users/sean/dev/tornado/data_streams/mixed_w_50_n_0.1/mixed_w_50_n_0.1_105.csv",
    "/Users/sean/dev/tornado/data_streams/mixed_w_50_n_0.1/mixed_w_50_n_0.1_104.csv",

    "/Users/sean/dev/tornado/data_streams/circles_w_500_n_0.1/circles_w_500_n_0.1_105.csv",
    "/Users/sean/dev/tornado/data_streams/circles_w_500_n_0.1/circles_w_500_n_0.1_104.csv",
    "/Users/sean/dev/tornado/data_streams/circles_w_500_n_0.1/circles_w_500_n_0.1_103.csv",
    "/Users/sean/dev/tornado/data_streams/circles_w_500_n_0.1/circles_w_500_n_0.1_102.csv",
    "/Users/sean/dev/tornado/data_streams/circles_w_500_n_0.1/circles_w_500_n_0.1_101.csv",

    "/Users/sean/dev/tornado/data_streams/led_w_500_n_0.1/led_w_500_n_0.1_104.csv",
    "/Users/sean/dev/tornado/data_streams/led_w_500_n_0.1/led_w_500_n_0.1_105.csv",
    "/Users/sean/dev/tornado/data_streams/led_w_500_n_0.1/led_w_500_n_0.1_101.csv",
    "/Users/sean/dev/tornado/data_streams/led_w_500_n_0.1/led_w_500_n_0.1_102.csv",
    "/Users/sean/dev/tornado/data_streams/led_w_500_n_0.1/led_w_500_n_0.1_103.csv",
    
    "/Users/sean/dev/tornado/data_streams/sine1_w_50_n_0.1/sine1_w_50_n_0.1_105.csv",
    "/Users/sean/dev/tornado/data_streams/sine1_w_50_n_0.1/sine1_w_50_n_0.1_104.csv",
    "/Users/sean/dev/tornado/data_streams/sine1_w_50_n_0.1/sine1_w_50_n_0.1_101.csv",
    "/Users/sean/dev/tornado/data_streams/sine1_w_50_n_0.1/sine1_w_50_n_0.1_103.csv",
    "/Users/sean/dev/tornado/data_streams/sine1_w_50_n_0.1/sine1_w_50_n_0.1_102.csv"
]

def thesis_experiment(window_type=SLIDING_TUMBLING, window_size=0):
    # for Tornado files
    for filepath in TORNADO_FILES:
        stream = FileStream(filepath=filepath)
        demo(stream, prepare_for_use(stream, True), batch_size=window_size)
    
    # for SEA generator
    for noise_percentage in range(0.0, 0.7, 0.1):
        stream = SEAGenerator(noise_percentage=noise_percentage)
        demo(stream, prepare_for_use(stream, False), batch_size=window_size)
    
    # for Waveform generator
    stream = WaveformGenerator(has_noise=True)
    demo(stream, prepare_for_use(stream, False), batch_size=window_size)

def prepare_for_use(stream, file_stream):
    stream.prepare_for_use()
    return np.asarray(
        stream.get_target_values() if file_stream else stream.target_values)

## Examine how sliding windows perform against tumbling windows and against sliding tumbling windows
thesis_experiment(window_type=SLIDING)
thesis_experiment(window_type=TUMBLING)
thesis_experiment(window_type=SLIDING_TUMBLING)

## The size of the window or batch (w) will have an impact on the results; we probably need some experiments about that.
for window_size in range(0, 100, 10):
    thesis_experiment(window_size=window_size)

## Compare different voting ensemble strategies against one another and against single classifiers and against other ensemble methods. Compare outcomes

## See how the modified concept drift detector performs with/without sliding tumbling windows, and/or when playing with the ensemble classifier reset logic, and finding the right balance of ground truth that can be omitted versus using predicted values as the ground truth

## Evaluate the performance, stream velocity, accuracy against other methods

## Determine if the summarising classifiers improve performance

## Determine threshold when best to use summarizer over the normal voting classifiers.

## It is good to compare to blind adaptation, i.e. a simple model reset at every x instances.

## Compare to OzaBoost and OzaBag.