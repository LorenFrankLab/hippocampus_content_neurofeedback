# required packages:
import datajoint as dj
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

from spyglass.common import (
    IntervalPositionInfo,
    IntervalList,
    StateScriptFile,
    DIOEvents,
    SampleCount,
)
from spyglass.utils.dj_helper_fn import fetch_nwb
from hippocampus_content_neurofeedback.mcoulter_realtime_filename import (
    RealtimeFilename,
)

import pprint

import warnings

# import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import logging
import datetime
import statistics
import os

FORMAT = "%(asctime)s %(message)s"

logging.basicConfig(level="INFO", format=FORMAT, datefmt="%d-%b-%y %H:%M:%S")
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

schema = dj.schema("mcoulter_reward_spectogram")


# parameters
@schema
class RewardSpectogramParameters(dj.Manual):
    definition = """
    realtime_error_param_name : varchar(500)
    ---
    realtime_error_parameters : blob
    """

    # can include description above
    # description : varchar(500)

    def insert_default(self):
        realtime_error_parameters = {}
        realtime_error_parameters["center_well_bin"] = 3
        realtime_error_parameters["in_out"] = 1
        realtime_error_parameters["number_of_runs"] = 3
        realtime_error_parameters["number_of_bins"] = 3
        realtime_error_parameters["velocity_filter"] = 10
        realtime_error_parameters[
            "file_directory"
        ] = "/cumulus/mcoulter/tonks/realtime_merged/"
        # realtime_error_parameters['ymin'] = 30
        # realtime_error_parameters['ymax'] = 160
        self.insert1(
            ["default", realtime_error_parameters], skip_duplicates=True
        )


# i dont think we need samplecount or statescriptfile in this list
# select parameters and cluster
@schema
class RewardSpectogramSelection(dj.Manual):
    definition = """
    -> RealtimeFilename
    -> RewardSpectogramParameters
    ---
    """


# this is the computed table - basically it has the results of the analysis
@schema
class RewardSpectogram(dj.Computed):
    definition = """
    -> RewardSpectogramSelection
    ---
    error_table : blob

    """

    def make(self, key):
        print(f"Computing realtime error for: {key}")

        if (
            key["nwb_file_name"].split("2")[0]
            == key["realtime_filename"].split("_")[3]
        ):
            print("names match")
        elif (
            key["nwb_file_name"].split("2")[0]
            == key["realtime_filename"].split("_")[2]
        ):
            print("names match")
        else:
            raise ValueError("rat names dont match")

        # these are all in the paramters set now
        error_parameters = (
            RewardSpectogramParameters
            & {"realtime_error_param_name": key["realtime_error_param_name"]}
        ).fetch()[0][1]
        center_well_bin = error_parameters["center_well_bin"]

        # NOTE: for in_out: 1 is both direction, 2 is outbound, 3 is inbound
        in_out = error_parameters["in_out"]

        number_of_runs = error_parameters["number_of_runs"]
        number_of_bins = error_parameters["number_of_bins"]
        velocity_filter = error_parameters["velocity_filter"]
        file_directory = error_parameters["file_directory"]

        # ginny/tonks/molly/arthur: center well is at position 4
        # ron/george and pippin: center well is at position 3
        # center_well_bin = 3

        # array to hold results: position x arm visit
        posterior_fraction_array = np.zeros((25, 8))

        # comment out if just using 1 hdf file
        hdf_file = file_directory + key["realtime_filename"]

        store = pd.HDFStore(hdf_file, mode="r")

        decoder_data = pd.read_hdf(hdf_file, key="rec_4")
        occupancy_data = pd.read_hdf(hdf_file, key="rec_7")

        # encoder_data = pd.read_hdf(hdf_file,key='rec_3')
        # decoder_data = pd.read_hdf(hdf_file,key='rec_4')
        # likelihood_data = pd.read_hdf(hdf_file,key='rec_6')
        ##decoder_missed_spikes = store['rec_5']
        ##ripple_data = store['rec_1']
        ##stim_state = store['rec_10']
        # stim_lockout = pd.read_hdf(hdf_file,key='rec_11')
        # stim_message = pd.read_hdf(hdf_file,key='rec_12')
        # occupancy_data = pd.read_hdf(hdf_file,key='rec_7')

        print(hdf_file)

        # here is where all of the code will go from the jupyter notebook

        key["error_table"] = posterior_fraction_array
        self.insert1(key)
