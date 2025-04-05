
import ast
import numpy as np

import yaml, os
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from bci_plot.utils import eeg_stats

def gen_adaptation_stats(session):
    session_stats = eeg_stats.get_stats(session, use_ftt=True)
    return session_stats
