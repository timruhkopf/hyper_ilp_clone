#!/bin/bash

# Random search HPO on StarE run
# with a considerably smaller budget (a 10th)
#ilp tune --model stare --num-epochs=100
ilp run smac_stare -wb=True -hp=rs -ne=40

# SMAC applied to StarE run
ilp run smac_stare -wb=True -hp=smac -ne=40
