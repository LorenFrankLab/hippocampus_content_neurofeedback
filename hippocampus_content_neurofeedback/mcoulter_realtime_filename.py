# required packages:
import datajoint as dj
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

from spyglass.common import IntervalPositionInfo, IntervalList, StateScriptFile, DIOEvents, SampleCount
from spyglass.common.dj_helper_fn import fetch_nwb

import pprint

import warnings
#import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import logging
import datetime
#from pynwb import NWBHDF5IO, NWBFile

FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(level='INFO', format=FORMAT, datefmt='%d-%b-%y %H:%M:%S')
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=ResourceWarning)

schema = dj.schema("mcoulter_realtime_filename")

@schema
class RealtimeFilename(dj.Manual):
    definition = """
    -> IntervalList
    realtime_filename: varchar (255)
    ---
    """
