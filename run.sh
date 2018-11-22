#!/bin/bash

# python3 setup.py install
rm -r experiment_results/ ; pip3 install -U . && python3 src/skmultiflow/demos/thesis_experiments.py
