'''
This is an iPython Notebook written in a python file for documentation purposes.
'''

%matplotlib inline

import numpy as np
import torch
from torch.autograd import Variable


from matplotlib import animation
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm, trange
from ipywidgets import interact, fixed

from sklearn.metrics import r2_score

from IPython.display import display, HTML