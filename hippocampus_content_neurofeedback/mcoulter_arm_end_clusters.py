# required packages:
import os
import datajoint as dj
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import pandas as pd
from ripple_detection import Kay_ripple_detector
from ripple_detection.detectors import get_Kay_ripple_consensus_trace
from scipy.stats import zscore
from spyglass.common.common_interval import interval_list_intersect
from hippocampus_content_neurofeedback.mcoulter_realtime_filename import (
    RealtimeFilename,
)
from spyglass.spikesorting.v0 import Waveforms, CuratedSpikeSorting
from spyglass.mcoulter_statescript_rewards import (
    StatescriptRewardSelection,
    StatescriptReward,
)
from spyglass.utils.dj_helper_fn import fetch_nwb
import pprint
import warnings
import numpy as np
import xarray as xr
import logging
import datetime
import scipy
from scipy import ndimage

import spyglass.common as sgc
from spyglass.lfp import LFPOutput
import spyglass.lfp.v1 as sglfp
import spyglass.lfp.analysis.v1 as lfp_analysis
from spyglass.position import PositionOutput
import spyglass.ripple.v1 as sgrip
from spyglass.lfp.v1 import LFPV1, LFPSelection
from spyglass.lfp.analysis.v1.lfp_band import LFPBandV1, LFPBandSelection
from spyglass.common import get_electrode_indices
from spyglass.decoding.v0.sorted_spikes import SortedSpikesIndicator

from spyglass.common import (
    IntervalPositionInfo,
    IntervalList,
    StateScriptFile,
    DIOEvents,
    SampleCount,
)

FORMAT = "%(asctime)s %(message)s"

logging.basicConfig(level="INFO", format=FORMAT, datefmt="%d-%b-%y %H:%M:%S")
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

schema = dj.schema("mcoulter_arm_end_clusters")


# parameters to calculate place field
@schema
class ArmEndClustersParameters(dj.Manual):
    definition = """
    arm_end_clusters_param_name : varchar(500)
    ---
    arm_end_clusters_parameters : blob
    """

    # can include description above
    # description : varchar(500)

    def insert_default(self):
        arm_end_clusters_parameters = {}
        arm_end_clusters_parameters["well_distance_max"] = 17
        arm_end_clusters_parameters["center_well_pos"] = [449, 330]
        arm_end_clusters_parameters["arm_posterior_fraction"] = 0.4
        arm_end_clusters_parameters["center_posterior_fraction"] = 0.6
        arm_end_clusters_parameters[
            "file_directory"
        ] = "/cumulus/mcoulter/tonks/realtime_merged/"
        self.insert1(
            ["default", arm_end_clusters_parameters], skip_duplicates=True
        )


# select parameters
@schema
class ArmEndClustersSelection(dj.Manual):
    definition = """
    -> RealtimeFilename
    -> ArmEndClustersParameters
    ---
    """


# this is the computed table - basically it has the results of the analysis
@schema
class ArmEndClusters(dj.Computed):
    definition = """
    -> ArmEndClustersSelection
    ---
    arm1_end_spikes : mediumblob
    arm2_end_spikes : mediumblob
    cluster_id : mediumblob
    """

    def make(self, key):
        print(f"Computing LFP posterior sum for: {key}")

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        def find_nearest_1(array, value):
            array = np.asarray(array)
            diff_array = np.abs(array - value)
            idx = diff_array.argmin()
            diff_min = diff_array[idx]
            return array[idx], idx, diff_min


        # StateScriptFile: get the time slice for taskState1
        # this should all come from the key
        # nwb_file_name = 'ginny20211101_.nwb'

        # need to check interval_list_name and nwb_file_name and include all possibilites here

        # insert code here

        arm_end_clusters_parameters = (
            ArmEndClustersParameters
            & {
                "arm_end_clusters_param_name": key[
                    "arm_end_clusters_param_name"
                ]
            }
        ).fetch()[0][1]

        post_thresh = arm_end_clusters_parameters["arm_posterior_fraction"]
        well_distance_max = arm_end_clusters_parameters["well_distance_max"]

        # add new parameters
        # smoothing: yes/no
        # time overlap with ripples
        #smoothing_on = arm_end_clusters_parameters['smoothing']
        #ripple_prox = arm_end_clusters_parameters['rip_prox']


        rat_name = key["nwb_file_name"].split("2")[0]

        # how to get realtime filename - not needed
        if rat_name == 'ron':
           path_add = "/cumulus/mcoulter/george/realtime_rec_merged/"
        elif rat_name == 'tonks':
           path_add = "/cumulus/mcoulter/tonks/realtime_merged/"
        elif rat_name == 'arthur':
           path_add = "/cumulus/mcoulter/molly/realtime_rec_merged/"
        elif rat_name == 'molly':
           path_add = "/cumulus/mcoulter/molly/realtime_rec_merged/"
        elif rat_name == 'ginny':
           path_add = "/cumulus/mcoulter/tonks/realtime_merged/"
        elif rat_name == 'pippin':
           path_add = "/cumulus/mcoulter/pippin/realtime_rec_merged_end_all/"

        #path_add = lfp_posterior_sum_parameters["file_directory"]

        # set up variables used below
        arm1_all_tet_spike_count = []
        arm2_all_tet_spike_count = []
        arm1_all_tet_cluster_ids = []
        arm2_all_tet_cluster_ids = []
           
        # get pos name
        pos_interval_list = (
            StatescriptReward & {"nwb_file_name": key["nwb_file_name"]}
        ).fetch("interval_list_name")
        realtime_interval_list = (
            RealtimeFilename & {"nwb_file_name": key["nwb_file_name"]}
        ).fetch("interval_list_name")
        print(realtime_interval_list)

        # set up variables used below
        arm1_all_tet_spike_count = []
        arm2_all_tet_spike_count = []
        all_tet_cluster_ids = []
        
        # exception for molly 3-24
        try:
            if key["interval_list_name"] == realtime_interval_list[0]:
                if key["nwb_file_name"] == "molly20220324_.nwb":
                    pos_name = 'pos 1 valid times'
                else:
                    pos_name = pos_interval_list[0]
            elif key["interval_list_name"] == realtime_interval_list[1]:
                if key["nwb_file_name"] == "molly20220324_.nwb":
                    pos_name = 'pos 3 valid times'
                else:
                    pos_name = pos_interval_list[1]
            elif key["interval_list_name"] == realtime_interval_list[2]:
                if key["nwb_file_name"] == "molly20220324_.nwb":
                    pos_name = 'pos 6 valid times'
                else:
                    pos_name = pos_interval_list[2]
            print(key["nwb_file_name"], pos_name, key["interval_list_name"])
            pos_name_good = 1
        except IndexError as e:
            print('missing pos name. skip',key["nwb_file_name"] )
            pos_name_good = 0

        if pos_name_good == 1:
            # load real-time recording file
            os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
            hdf_file = path_add + key["realtime_filename"]

            # print(f'Computing realtime posterior sum for: {key}')

            # store = pd.HDFStore(hdf_file, mode='r')

            decoder_data = pd.read_hdf(hdf_file, key="rec_4")
            occupancy_data = pd.read_hdf(hdf_file, key="rec_7")
            encoder_data = pd.read_hdf(hdf_file, key="rec_3")
            # likelihood_data = pd.read_hdf(hdf_file,key='rec_6')
            ##decoder_missed_spikes = store['rec_5']
            ##ripple_data = store['rec_1']
            ##stim_state = store['rec_10']
            stim_lockout = pd.read_hdf(hdf_file, key="rec_11")
            stim_message = pd.read_hdf(hdf_file, key="rec_12")

            print(hdf_file)
            print(encoder_data.shape)
            # print(encoder_data[encoder_data['encode_spike']==1].shape)

            # reward count
            # reward_counts = stim_message[stim_message['shortcut_message_sent']==1].shape[0]
            # print('rewards',stim_message[stim_message['shortcut_message_sent']==1].shape[0])
            ##print('instructive rewards',np.count_nonzero(reward_list[i_session,:]))

            # reward count - lockout removed
            stim_message_diff = stim_message.copy()
            stim_message_diff_reward = stim_message_diff[
                (stim_message_diff["shortcut_message_sent"] == 1)
            ]
            stim_message_diff_reward["timestamp_diff"] = stim_message_diff_reward[
                "bin_timestamp"
            ].diff()
            stim_message_diff_reward_2 = pd.concat(
                [
                    stim_message_diff_reward[0:1],
                    stim_message_diff_reward[
                        stim_message_diff_reward["timestamp_diff"] > 30000
                    ],
                ]
            )
            print("rewards", stim_message_diff_reward_2.shape)

            # only high reward sessions
            if stim_message_diff_reward_2.shape[0] > 0:

                # fred, pippin, percy
                if rat_name == "pippin":
                    center_well_pos = [634, 648]
                else:
                    # geroge, ron, ginny, tonks
                    center_well_pos = [449, 330]

                decoder_data_center_well = decoder_data.copy()

                # gaussian smooth posterior and then add column for arm1_end_smooth and arm2_end_smooth
                # much more smoothing: try 20 with 0.2
                gauss = pd.DataFrame(ndimage.gaussian_filter1d(decoder_data_center_well.iloc[:,27:68], 10, 0))
                decoder_data_center_well['arm1_end_smooth'] = (gauss[20]+gauss[21]+gauss[22]+gauss[23]+gauss[24])>0.2
                decoder_data_center_well['arm2_end_smooth'] = (gauss[36]+gauss[37]+gauss[38]+gauss[39]+gauss[40])>0.2

                decoder_data_center_well["center_well_dist"] = (
                    np.sqrt(
                        np.square(
                            decoder_data_center_well["smooth_x"] - center_well_pos[0]
                        )
                        + np.square(
                            decoder_data_center_well["smooth_y"] - center_well_pos[1]
                        )
                    )
                    * 0.222
                )
                decoder_data_center_well["arm1_end"] = (
                    decoder_data_center_well["x20"]
                    + decoder_data_center_well["x21"]
                    + decoder_data_center_well["x22"]
                    + decoder_data_center_well["x23"]
                    + decoder_data_center_well["x24"]
                )
                decoder_data_center_well["arm2_end"] = (
                    decoder_data_center_well["x36"]
                    + decoder_data_center_well["x37"]
                    + decoder_data_center_well["x38"]
                    + decoder_data_center_well["x39"]
                    + decoder_data_center_well["x40"]
                )

                decoder_data_center_well["arm1_base"] = (
                    decoder_data_center_well["x13"]
                    + decoder_data_center_well["x14"]
                    + decoder_data_center_well["x15"]
                    + decoder_data_center_well["x16"]
                    + decoder_data_center_well["x17"]
                )
                decoder_data_center_well["arm2_base"] = (
                    decoder_data_center_well["x29"]
                    + decoder_data_center_well["x30"]
                    + decoder_data_center_well["x31"]
                    + decoder_data_center_well["x32"]
                    + decoder_data_center_well["x33"]
                )

                decoder_data_center_well["arm1_all"] = (
                    decoder_data_center_well["x13"]
                    + decoder_data_center_well["x14"]
                    + decoder_data_center_well["x15"]
                    + decoder_data_center_well["x16"]
                    + decoder_data_center_well["x17"]
                    + decoder_data_center_well["x18"]
                    + decoder_data_center_well["x19"]
                    + decoder_data_center_well["x20"]
                    + decoder_data_center_well["x21"]
                    + decoder_data_center_well["x22"]
                    + decoder_data_center_well["x23"]
                    + decoder_data_center_well["x24"]
                )
                decoder_data_center_well["arm2_all"] = (
                    decoder_data_center_well["x29"]
                    + decoder_data_center_well["x30"]
                    + decoder_data_center_well["x31"]
                    + decoder_data_center_well["x32"]
                    + decoder_data_center_well["x33"]
                    + decoder_data_center_well["x34"]
                    + decoder_data_center_well["x35"]
                    + decoder_data_center_well["x36"]
                    + decoder_data_center_well["x37"]
                    + decoder_data_center_well["x38"]
                    + decoder_data_center_well["x39"]
                    + decoder_data_center_well["x40"]
                )

                decoder_data_center_well["center_decode"] = (
                    decoder_data_center_well["x00"]
                    + decoder_data_center_well["x01"]
                    + decoder_data_center_well["x02"]
                    + decoder_data_center_well["x03"]
                    + decoder_data_center_well["x04"]
                    + decoder_data_center_well["x05"]
                    + decoder_data_center_well["x06"]
                    + decoder_data_center_well["x07"]
                    + decoder_data_center_well["x08"]
                )

                task2_decode_center = decoder_data_center_well[
                    (decoder_data_center_well["center_well_dist"] < well_distance_max)
                    & (decoder_data_center_well["taskState"] == 2)
                ]

                # # add binary column for smoothed arm ends, then get intervals
                # task2_decode_center['group_arm1'] = (task2_decode_center['arm1_end_smooth'] != 
                #                                         task2_decode_center['arm1_end_smooth'].shift()).cumsum()
                # arm1_intervals = (task2_decode_center[task2_decode_center['arm1_end_smooth'] == 1].
                #                     groupby('group_arm1').apply(lambda x: (x.index[0], x.index[-1])).tolist())
                # task2_decode_center['group_arm2'] = (task2_decode_center['arm2_end_smooth'] != 
                #                                         task2_decode_center['arm2_end_smooth'].shift()).cumsum()
                # arm2_intervals = (task2_decode_center[task2_decode_center['arm2_end_smooth'] == 1].
                #                     groupby('group_arm2').apply(lambda x: (x.index[0], x.index[-1])).tolist())
                # print('arm1 end intervals',len(arm1_intervals),'arm2 end intervals',len(arm2_intervals))
                # # get timestamp for center bin
                # arm1_interval_mid_list = []
                # arm2_interval_mid_list = []
                # for event in arm1_intervals:
                #     try:
                #         midpoint_event = (event[1] - round((event[1]-event[0])/2))
                #         mid_timestamp = task2_decode_center.loc[[midpoint_event]]['bin_timestamp'].values[0]
                #         arm1_interval_mid_list.append(mid_timestamp)
                #     except KeyError as e:
                #         print('event did not match')
                # for event in arm2_intervals:
                #     try:
                #         midpoint_event = (event[1] - round((event[1]-event[0])/2))
                #         mid_timestamp = task2_decode_center.loc[[midpoint_event]]['bin_timestamp'].values[0]
                #         arm2_interval_mid_list.append(mid_timestamp)
                #     except KeyError as e:
                #         print('event did not match')
                # arm1_smooth_midpoint = np.array(arm1_interval_mid_list)
                # arm2_smooth_midpoint = np.array(arm2_interval_mid_list)

                # task2_head_dir = pd.merge_asof(
                #    task2_decode_center,
                #    occupancy_data,
                #    on="bin_timestamp",
                #    direction="nearest",
                # )
                # print('timebins task2 near center',task2_head_dir.shape)

                # task2_head_dir_reward = pd.merge_asof(task2_head_dir,stim_message_diff_reward_2,on='bin_timestamp',
                #                                      direction='nearest',tolerance=50)

                # reward rate (per min): reward count / time at center well
                reward_rate = (
                    (stim_message_diff_reward_2.shape[0] / task2_decode_center.shape[0])
                    * (1000 / 6)
                    * 60
                )

                # need this for skipped sessions
                overlap_list = []
                overlap_array = np.array(overlap_list)
                all_overlap_arm1 = overlap_array.copy()
                all_overlap_arm2 = overlap_array.copy()
                all_overlap_reward = overlap_array.copy()

                if task2_decode_center.shape[0] > 0:
                    # if task2_decode_center.shape[0] > 0 and key["nwb_file_name"] != 'arthur20220314_.nwb' and key["nwb_file_name"] != 'molly20220315_.nwb':

                    print(
                        ">0.4 arm1 end",
                        task2_decode_center[
                            task2_decode_center["arm1_end"] > post_thresh
                        ].shape[0]
                        / task2_decode_center.shape[0],
                    )
                    print(
                        ">0.4 arm2 end",
                        task2_decode_center[
                            task2_decode_center["arm2_end"] > post_thresh
                        ].shape[0]
                        / task2_decode_center.shape[0],
                    )
                    arm1_end = (
                        task2_decode_center[
                            task2_decode_center["arm1_end"] > post_thresh
                        ].shape[0]
                        / task2_decode_center.shape[0]
                    )
                    arm2_end = (
                        task2_decode_center[
                            task2_decode_center["arm2_end"] > post_thresh
                        ].shape[0]
                        / task2_decode_center.shape[0]
                    )

                    print(
                        ">0.4 arm1 base",
                        task2_decode_center[
                            task2_decode_center["arm1_base"] > post_thresh
                        ].shape[0]
                        / task2_decode_center.shape[0],
                    )
                    print(
                        ">0.4 arm2 base",
                        task2_decode_center[
                            task2_decode_center["arm2_base"] > post_thresh
                        ].shape[0]
                        / task2_decode_center.shape[0],
                    )
                    arm1_base = (
                        task2_decode_center[
                            task2_decode_center["arm1_base"] > post_thresh
                        ].shape[0]
                        / task2_decode_center.shape[0]
                    )
                    arm2_base = (
                        task2_decode_center[
                            task2_decode_center["arm2_base"] > post_thresh
                        ].shape[0]
                        / task2_decode_center.shape[0]
                    )
                    print("all task2 center bins", task2_decode_center.shape)

                    arm1_end_timebins = task2_decode_center[
                        task2_decode_center["arm1_end"] > post_thresh
                    ]["bin_timestamp"].values
                    arm2_end_timebins = task2_decode_center[
                        task2_decode_center["arm2_end"] > post_thresh
                    ]["bin_timestamp"].values

                    average_arm_end_bins = np.int(
                        np.mean(
                            [arm1_end_timebins.shape[0], arm1_end_timebins.shape[0]]
                        )
                    )

                    center_well_timebins = task2_decode_center[
                        task2_decode_center["center_decode"] > 0.6
                    ]["bin_timestamp"].values
                    center_well_timebins_sub = np.random.choice(
                        center_well_timebins, average_arm_end_bins
                    )
                    print(
                        "subsample center bins",
                        average_arm_end_bins,
                        center_well_timebins_sub.shape,
                    )

                    # want to use new time boundaries to look for overlap with arm end bins
                    # do this below and just assume each arm end bin is 6msec long
                    # could make the arm end bins into interval list
                    # then find intersection times and then sum intersection times
                    # the divide this by total time
                    # hmm we actually just want overlap - fraction of event during ripple time, so use as it below

                    # ripples and calculate overlap
                    # pos_name = key["interval_list_name"]

                    # run_list = (sgrip.RippleTimesV1() & {'nwb_file_name':key["nwb_file_name"]}).fetch('pos_merge_id')
                    # for run in run_list:

                    # pos_merge_id = (sgrip.RippleTimesV1() & {'nwb_file_name':key["nwb_file_name"]} &
                    #        {'pos_merge_id':run}).fetch('pos_merge_id')[filename_counter]
                    key_2 = {}
                    key_2["nwb_file_name"] = key["nwb_file_name"]
                    key_2["interval_list_name"] = pos_name
                    key_2["position_info_param_name"] = "default_decoding"

                    # use this info to find sort_interval - set first for pos_name 0,1 and second for 2
                    sort_intervals = np.unique(
                        (
                            Waveforms
                            & {"nwb_file_name": key["nwb_file_name"]}
                            & {"waveform_params_name": "default_whitened"}
                            & {'artifact_removed_interval_list_name LIKE "%100_prop%"'}
                        ).fetch("sort_interval_name")
                    )
                    print("sort intervals", sort_intervals)

                    #tet_list = np.unique(
                    #    (
                    #        Waveforms
                    #        & {"nwb_file_name": key["nwb_file_name"]}
                    #        & {"waveform_params_name": "default_whitened"}
                    #        & {'artifact_removed_interval_list_name LIKE "%100_prop%"'}
                    #    ).fetch("sort_group_id")
                    #)
                    #print("tetrodes", tet_list)

                    # try getting tet list from CuratedSpikeSorting.Unit to remove extra tetrodes first
                    tet_list = np.unique(
                        (
                            CuratedSpikeSorting.Unit
                            & {"nwb_file_name": key["nwb_file_name"]}
                            & {"sorter": "mountainsort4"}
                            & {'artifact_removed_interval_list_name LIKE "%100_prop%"'}
                        ).fetch("sort_group_id")
                    )
                    print("tetrodes", tet_list)                    

                    tet_list_3 = np.unique((CuratedSpikeSorting & {"nwb_file_name": key["nwb_file_name"]}
                                        & {'curation_id':3}).fetch('sort_group_id'))

                    all_pos = np.unique(
                        (
                            StatescriptReward & {"nwb_file_name": key["nwb_file_name"]}
                        ).fetch("interval_list_name")
                    )
                    if pos_name == all_pos[0]:
                        sort_interval = sort_intervals[0]
                    elif pos_name == all_pos[1]:
                        sort_interval = sort_intervals[0]
                    elif pos_name == all_pos[2]:
                        sort_interval = sort_intervals[1]

                    # need to convert realtime timestamps to spyglass
                    offset = (
                        StatescriptReward()
                        & {"nwb_file_name": key["nwb_file_name"]}
                        & {"interval_list_name": pos_name}
                    ).fetch("offset")

                    # need to loop through tetrodes
                    # note: this works to get all clusters because all columns will still be in dataframe even if only spiking in a few
                    
                    tet_counter = 0
                    for tetrode in tet_list:
                        # need sortedspikesindicator
                        if tetrode in tet_list_3:
                            print('manually curated tetrode')
                            try:
                                sorted_spikes_table = (SortedSpikesIndicator & {'nwb_file_name' : key["nwb_file_name"]}
                                                & {'artifact_removed_interval_list_name LIKE "%100_prop%"'}
                                                & {'sort_interval_name': sort_interval}
                                                & {'interval_list_name':pos_name} & {'curation_id':3}
                                                    & {'sort_group_id':tetrode}).fetch_dataframe() 
                                spike_table_good = 1
                            except ValueError as e:
                                print('missing tet. other sort interval',tetrode)
                                spike_table_good = 0
                        else:   
                            try:                         
                                sorted_spikes_table = (SortedSpikesIndicator & {'nwb_file_name' : key["nwb_file_name"]}
                                                & {'artifact_removed_interval_list_name LIKE "%100_prop%"'}
                                                & {'sort_interval_name': sort_interval}
                                                & {'interval_list_name':pos_name} & {'curation_id':1}
                                                    & {'sort_group_id':tetrode}).fetch_dataframe() 
                                spike_table_good = 1
                            except ValueError as e:
                                print('missing tet. other sort interval',tetrode)
                                spike_table_good = 0      

                        if spike_table_good == 1:                     
                            spike_timestamps = sorted_spikes_table.index

                            # note: based on code above, timebin from realtime system is start of 6ms timebin
                            # original
                            time_loop_counter = 0
                            for time in arm1_end_timebins:
                            # smoothed arm end events
                            #for time in arm1_smooth_midpoint:
                                adjusted_timestamp = time/30000 + offset

                                # NOTE: should only use if this is a close match 
                                rip_time,placeholder,min_diff = find_nearest_1(spike_timestamps,adjusted_timestamp)
                                # need to find closest timestamp
                                if min_diff < 0.001:  
                                    spike_count_single = sorted_spikes_table.loc[rip_time:rip_time+0.0059].sum(axis=0)
                                    if time_loop_counter == 0:
                                        print(rip_time,min_diff)
                                        spike_count_array = spike_count_single.copy()
                                    else:
                                        spike_count_array += spike_count_single
                                    time_loop_counter += 1
                                else:
                                    print('spike time mismatch')
                            print('arm 1. tetrode',tetrode)
                            #print(spike_count_array)
                            
                            # need to pull out cluster IDs
                            #reward_cluster_out_list.append([np.int(item[0].split('_')[1]),
                            #                                np.int(item[0].split('_')[2])])
                            if tet_counter == 0:
                                arm1_all_tet_spike_count = spike_count_array.values
                                all_tet_cluster_ids = np.array(spike_count_array.index)
                            else:
                                arm1_all_tet_spike_count = np.concatenate((arm1_all_tet_spike_count,
                                                                        spike_count_array.values),axis=0)
                                all_tet_cluster_ids = np.concatenate((all_tet_cluster_ids,
                                                                        spike_count_array.index),axis=0)
                            
                            time_loop_counter = 0
                            for time in arm2_end_timebins:
                            # smoothed arm end events
                            #for time in arm2_smooth_midpoint:
                                adjusted_timestamp = time/30000 + offset

                                # NOTE: should only use if this is a close match 
                                rip_time,placeholder,min_diff = find_nearest_1(spike_timestamps,adjusted_timestamp)
                                # need to find closest timestamp
                                if min_diff < 0.001:  
                                    spike_count_single = sorted_spikes_table.loc[rip_time:rip_time+0.0059].sum(axis=0)
                                    if time_loop_counter == 0:
                                        spike_count_array = spike_count_single.copy()
                                    else:
                                        spike_count_array += spike_count_single
                                    time_loop_counter += 1
                                else:
                                    print('spike time mismatch')
                                
                            print('arm 2. tetrode',tetrode)
                            #print(spike_count_array)
                            
                            if tet_counter == 0:
                                arm2_all_tet_spike_count = spike_count_array.values
                                #all_tet_cluster_ids = np.array(spike_count_array.index)
                            else:
                                arm2_all_tet_spike_count = np.concatenate((arm2_all_tet_spike_count,
                                                                        spike_count_array.values),axis=0)
                                #all_tet_cluster_ids = np.concatenate((all_tet_cluster_ids,
                                #                                        spike_count_array.index),axis=0)
                                                                
                            # this can check that the two intervals are lined up
                            #plt.figure()
                            #plt.scatter(target_arm_end_df['start'],np.repeat(1,target_arm_end_df.shape[0]))
                            #plt.scatter(ripples_df['start'],np.repeat(2,ripples_df.shape[0]))
                            tet_counter += 1


                            # this can check that the two intervals are lined up
                            # plt.figure()
                            # plt.scatter(target_arm_end_df['start'],np.repeat(1,target_arm_end_df.shape[0]))
                            # plt.scatter(ripples_df['start'],np.repeat(2,ripples_df.shape[0]))

                            #except ValueError as e:
                            #    print('no spikes on tetrode',tetrode)
                            
            print('arm 2',arm2_all_tet_spike_count)
            print('arm 1',arm1_all_tet_spike_count)

            print(key['nwb_file_name'],arm1_all_tet_spike_count,arm2_all_tet_spike_count,all_tet_cluster_ids)

            key["arm1_end_spikes"] = arm1_all_tet_spike_count
            key["arm2_end_spikes"] = arm2_all_tet_spike_count
            key["cluster_id"] = all_tet_cluster_ids


            self.insert1(key)
