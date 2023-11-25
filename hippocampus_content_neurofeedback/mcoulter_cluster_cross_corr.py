# required packages:
import datajoint as dj
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from scipy import ndimage
from scipy.ndimage import gaussian_filter

from spyglass.decoding import SortedSpikesIndicator
from spyglass.common import IntervalPositionInfo
from spyglass.common import StateScriptFile, SampleCount, Nwbfile
from spyglass.utils.dj_helper_fn import fetch_nwb

from spyglass.common.common_position import TrackGraph
from spyglass.common.common_nwbfile import AnalysisNwbfile

from hippocampus_content_neurofeedback.mcoulter_reward_cluster_list import (
    RewardClusterList,
)
import pprint

import warnings
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import logging
import datetime
from pynwb import NWBHDF5IO, NWBFile
from ndx_xarray import ExternalXarrayDataset
from itertools import product

FORMAT = "%(asctime)s %(message)s"

logging.basicConfig(level="INFO", format=FORMAT, datefmt="%d-%b-%y %H:%M:%S")
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)

schema = dj.schema("mcoulter_cluster_cross_corr")


# parameters
@schema
class CrossCorrelationParam(dj.Manual):
    definition = """
    cross_corr_param_name : varchar(500)
    ---
    cross_corr_parameters : mediumblob
    """

    # can include description above
    # description : varchar(500)

    def insert_default(self):
        parameters = {}
        parameters["max_clusters"] = 40
        parameters["max_dt"] = 0.5
        parameters["min_dt"] = 0
        parameters["smoothing_step"] = 0.005
        self.insert1(["default", parameters], skip_duplicates=True)


# select parameters and cluster
@schema
class CrossCorrelationSelection(dj.Manual):
    definition = """
    -> RewardClusterList
    -> CrossCorrelationParam
    ---
    """


# this is the computed table - basically it has the results of the analysis
@schema
class CrossCorrelation(dj.Computed):
    definition = """
    -> CrossCorrelationSelection
    ---
    max_time : blob
    max_peak : blob
    """
    # other way: place_field : mediumblob

    def make(self, key):
        print(f"Computing cross correlation for: {key}")
        # this is where all the calculations go to generate the place field

        def _compute_correlogram(spk_times_1, spk_times_2, max_dt=None, min_dt=None):
            # Get inputs if not passed
            time_diff = (
                np.tile(spk_times_1, (spk_times_2.size, 1)) - spk_times_2[:, np.newaxis]
            )
            ind = np.logical_and(
                np.abs(time_diff) > min_dt, np.abs(time_diff) <= max_dt
            )
            time_diff = np.sort(time_diff[ind])
            return time_diff

        cross_corr_parameters = (
            CrossCorrelationParam
            & {"cross_corr_param_name": key["cross_corr_param_name"]}
        ).fetch()[0][1]

        max_clusters = cross_corr_parameters["max_clusters"]

        reward_cross_corr_max_time = []
        reward_cross_corr_max_peak = []

        print(key["nwb_file_name"])
        last_pos_interval = []
        try:
            reward_cluster_dict = (
                RewardClusterList()
                & {"nwb_file_name": key["nwb_file_name"]}
                & {
                    "reward_cluster_list_param_name": key[
                        "reward_cluster_list_param_name"
                    ]
                }
            ).fetch("reward_cluster_dict")
            sort_interval_list = []
            for item in reward_cluster_dict[0][key["nwb_file_name"]].items():
                # print(item[0])
                sort_interval_list.append(item[0])

            for sort_interval in sort_interval_list:
                print(sort_interval)
                pos_interval_list = []
                for item in reward_cluster_dict[0][key["nwb_file_name"]][
                    sort_interval
                ].items():
                    # print(item[0])
                    pos_interval_list.append(item[0])

                for pos_interval in pos_interval_list:
                    print(pos_interval)
                    if pos_interval == last_pos_interval:
                        print("same interval, skip", last_pos_interval, pos_interval)
                    else:
                        spike_only_dict = {}
                        cluster_counter = 0

                        for item in reward_cluster_dict[0][key["nwb_file_name"]][
                            sort_interval
                        ][pos_interval].items():
                            # print('reward cluster',item[0])
                            # print('reward spikes',item[1][6])
                            spike_only_dict[cluster_counter] = item[1][6]
                            # print('spike count',len(spike_only_dict[cluster_counter]))
                            cluster_counter += 1

                        # summarize cross correlation

                        # for now: only do up to 40 clusters
                        if cluster_counter > max_clusters:
                            print("too many clusters, cap at 40")
                            cluster_counter = max_clusters

                        cluster_list = list(
                            product(np.arange(cluster_counter), repeat=2)
                        )
                        shorter_dict = set([tuple(sorted(i)) for i in cluster_list])
                        cross_corr_counter = 0

                        for item in shorter_dict:
                            # print(item[0],item[1])
                            if item[0] != item[1]:
                                # print(item)
                                cell_1 = spike_only_dict[item[0]]
                                cell_2 = spike_only_dict[item[1]]
                                cross_corr = _compute_correlogram(
                                    cell_1, cell_2, 0.5, 0
                                )

                                # smooth
                                hist_array = np.array(
                                    np.histogram(
                                        cross_corr,
                                        bins=np.arange(-0.5, 0.505, 0.005),
                                    )[0]
                                )
                                smooth_hist = ndimage.gaussian_filter1d(hist_array, 1)
                                reward_cross_corr_max_time.append(
                                    np.arange(-0.5, 0.5, 0.005)[np.argmax(smooth_hist)]
                                )
                                reward_cross_corr_max_peak.append(np.max(smooth_hist))

                                cross_corr_counter += 1
                        print("loop count", cross_corr_counter)
                    last_pos_interval = pos_interval

        except IndexError as e:
            print("no cluster data:", key["nwb_file_name"])

        # save cross corrlelation results
        print("number of correlations", len(reward_cross_corr_max_time))

        key["max_time"] = reward_cross_corr_max_time
        key["max_peak"] = reward_cross_corr_max_peak
        self.insert1(key)
