## In construction

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import scipy.optimize as opt

import numpy as np
import time
import os
import sys
import argparse
import ast

import MultipathLocationEstimator
import mpraylocsim as mpr

""" Para facer probas do simulador xeral de mpraylocsim e os cambios que lle aplique

All parameters currently modeling the ray simulator: 

    -N --noerror -S --minstd --maxstd -D -G --npathgeo --algs --label --nosave --nompg --noloc --show --print

 """

