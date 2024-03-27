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
from spyglass.spikesorting.v0 import Waveforms
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

schema = dj.schema("mcoulter_theta_phase_shuffles")


# parameters to calculate place field
@schema
class ThetaPhaseShufflesParameters(dj.Manual):
    definition = """
    theta_shuffle_param_name : varchar(500)
    ---
    theta_shuffle_parameters : blob
    """

    # can include description above
    # description : varchar(500)

    def insert_default(self):
        theta_shuffle_parameters = {}
        theta_shuffle_parameters["well_distance_max"] = 17
        theta_shuffle_parameters["center_well_pos"] = [0,0]
        theta_shuffle_parameters["arm_posterior_fraction"] = 0.4
        theta_shuffle_parameters["center_posterior_fraction"] = 0.6
        theta_shuffle_parameters[
            "file_directory"
        ] = "placeholder"
        self.insert1(
            ["default", theta_shuffle_parameters], skip_duplicates=True
        )

# select parameters
@schema
class ThetaPhaseShufflesSelection(dj.Manual):
    definition = """
    -> RealtimeFilename
    -> ThetaPhaseShufflesParameters
    ---
    """

# this is the computed table - basically it has the results of the analysis
@schema
class ThetaPhaseShuffles(dj.Computed):
    definition = """
    -> ThetaPhaseShufflesSelection
    ---
    target_arm : double

    theta_phase_random_array : mediumblob

    """

    def make(self, key):
        print(f"Computing theta phase shuffles for: {key}")

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

        table_parameters = (
            ThetaPhaseShufflesParameters
            & {
                "theta_shuffle_param_name": key[
                    "theta_shuffle_param_name"
                ]
            }
        ).fetch()[0][1]

        post_thresh = table_parameters["arm_posterior_fraction"]
        well_distance_max = table_parameters["well_distance_max"]

        theta_phase_random_array = np.zeros((1000,1000))

        reward_rate = 0

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

        # get pos name
        pos_interval_list = (
            StatescriptReward & {"nwb_file_name": key["nwb_file_name"]}
        ).fetch("interval_list_name")
        realtime_interval_list = (
            RealtimeFilename & {"nwb_file_name": key["nwb_file_name"]}
        ).fetch("interval_list_name")
        print(realtime_interval_list)

        # exception for molly 3-24
        try:
            if (
                key["nwb_file_name"] == "molly20220324_.nwb"
                and key["interval_list_name"] == realtime_interval_list[1]
            ):
                pos_name = pos_interval_list[1]

            elif key["interval_list_name"] == realtime_interval_list[0]:
                pos_name = pos_interval_list[0]
            elif key["interval_list_name"] == realtime_interval_list[1]:
                pos_name = pos_interval_list[1]
            elif key["interval_list_name"] == realtime_interval_list[2]:
                pos_name = pos_interval_list[2]
            print(key["nwb_file_name"], pos_name, key["interval_list_name"])
            pos_name_exist = 1
        except IndexError as e:
            print('pos name failed')
            pos_name_exist = 0

        if pos_name_exist == 1:
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

                task2_decode_center = decoder_data_center_well[
                    (decoder_data_center_well["center_well_dist"] < well_distance_max)
                    & (decoder_data_center_well["taskState"] == 2)
                ]

                # get timestamp offset
                offset = (StatescriptReward() & {'nwb_file_name':key["nwb_file_name"]} 
                                & {'interval_list_name':pos_name}).fetch('offset')

                # reward rate (per min): reward count / time at center well
                reward_rate = (
                    (stim_message_diff_reward_2.shape[0] / task2_decode_center.shape[0])
                    * (1000 / 6)
                    * 60
                )

                if task2_decode_center.shape[0] > 0:
                    # if key["nwb_file_name"] != 'arthur20220314_.nwb' and key["nwb_file_name"] != 'molly20220315_.nwb':

                    # lets just do 1000 times - need to start with 5000 to get ~1000 with speed >4

                    # random times
                    print('***random times***')
                    center_well_timebins_sub = np.zeros((1000,4000))
                    all_center_times = task2_decode_center['bin_timestamp'].values
                    for i in np.arange(1000):
                        center_well_timebins_sub[i,:] = np.random.choice(all_center_times, 4000)

                    center_well_timebins_adj = center_well_timebins_sub / 30000 + offset
                    print(
                        "1000x subsample center bins",
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

                    all_pos = np.unique(
                        (
                            StatescriptReward & {"nwb_file_name": key["nwb_file_name"]}
                        ).fetch("interval_list_name")
                    )
                    try:
                        if pos_name == all_pos[0]:
                            sort_interval = sort_intervals[0]
                        elif pos_name == all_pos[1]:
                            sort_interval = sort_intervals[0]
                        elif pos_name == all_pos[2]:
                            sort_interval = sort_intervals[1]

                        # get position and generate smooth speed
                        spyglass_position_df = (
                            IntervalPositionInfo
                            & {"nwb_file_name": key["nwb_file_name"]}
                            & {"interval_list_name": pos_name}
                            & {"position_info_param_name": "default_decoding"}
                        ).fetch1_dataframe()
                        spyglass_position_df[
                            "speed_smooth"
                        ] = scipy.ndimage.gaussian_filter1d(
                            spyglass_position_df["head_speed"], 2
                        )
                        pos_table_exist = 1
                    except (ValueError, IndexError) as e:
                        print('no position')
                        pos_table_exist = 0

                    if pos_table_exist == 1:
                        pos_merge_id = PositionOutput.merge_get_part(
                            restriction=key_2
                        ).fetch("merge_id")[0]


                        # need to convert realtime timestamps to spyglass
                        offset = (
                            StatescriptReward()
                            & {"nwb_file_name": key["nwb_file_name"]}
                            & {"interval_list_name": pos_name}
                        ).fetch("offset")


                        # calculate theta phase and theta power
                        # NOTE: for comparison for theta phase - need time with max decode at center well
                        # ref electrodes
                        # ginny: 112,149
                        # tonks: 44,220
                        # molly: 64,193
                        # arthur: 88,161
                        # ron: 109,148
                        if rat_name == "ron":
                            # the refs are far above HPC in ron, could try tet 57 (->56)
                            # ref_elec_list = [109,148]
                            ref_elec_list = [109, 224]
                            target_interval_name = (
                                key["nwb_file_name"]
                                + "_"
                                + sort_interval
                                + "_LFP_difference_600_frac_80_1000_frac_80_10ms_80_new_2_artifact_removed_valid_times"
                            )

                        # im not sure if we need this - looks like all days have no_ref valid times
                        # to get integer of date np.int(nwb_file_name.split('molly2022')[1].split('_')[0])
                        # elif rat_name == "molly" and dict_counter < 13:
                        #    ref_elec_list = [64, 193]
                        #    target_interval_name = (
                        #        key["nwb_file_name"]
                        #        + "_"
                        #        + sort_interval
                        #        + "_LFP_difference_600_frac_80_1000_frac_80_10ms_80_new_2_artifact_removed_valid_times"
                        #    )
                        elif rat_name == "molly":
                            ref_elec_list = [64, 193]
                            target_interval_name = (
                                key["nwb_file_name"]
                                + "_"
                                + sort_interval
                                + "_LFP_difference_600_frac_80_1000_frac_80_10ms_80_new_2_no_ref_artifact_removed_valid_times"
                            )

                        elif rat_name == "ginny":
                            ref_elec_list = [112, 149]
                            target_interval_name = (
                                key["nwb_file_name"]
                                + "_"
                                + sort_interval
                                + "_LFP_difference_600_frac_80_1000_frac_80_10ms_80_new_2_artifact_removed_valid_times"
                            )

                        elif rat_name == "tonks":
                            # ref 220 seems good (orig tet 56)
                            ref_elec_list = [44, 220]
                            target_interval_name = (
                                key["nwb_file_name"]
                                + "_"
                                + sort_interval
                                + "_LFP_difference_600_frac_80_1000_frac_80_10ms_80_new_2_artifact_removed_valid_times"
                            )

                        elif rat_name == "arthur":
                            ref_elec_list = [88, 161]
                            target_interval_name = (
                                key["nwb_file_name"]
                                + "_"
                                + sort_interval
                                + "_LFP_difference_600_frac_80_1000_frac_80_10ms_80_new_2_artifact_removed_valid_times"
                            )
                        elif rat_name == "pippin":
                            ref_elec_list = [120, 242]
                            target_interval_name = (
                                key["nwb_file_name"]
                                + "_"
                                + sort_interval
                                + "_LFP_difference_600_frac_80_1000_frac_80_10ms_80_new_2_no_ref_artifact_removed_valid_times"
                            )

                        try:
                            theta_phase_df = (
                                LFPBandV1()
                                & {"nwb_file_name": key["nwb_file_name"]}
                                & {"filter_name": "Theta 5-11 Hz"}
                                & {"target_interval_list_name": target_interval_name}
                            ).compute_signal_phase(ref_elec_list)
                            theta_phase_exist = 1
                        except IndexError as e:
                            print('theta phase failed')
                            theta_phase_exist = 0

                        if theta_phase_exist == 1:
                            # get LFP timestamps from LFPBand table
                            lfp_data = (
                                LFPBandV1()
                                & {"nwb_file_name": key["nwb_file_name"]}
                                & {"filter_name": "Theta 5-11 Hz"}
                                & {"target_interval_list_name": target_interval_name}).fetch_nwb()
                            
                            lfp_timestamps = np.array(lfp_data[0]['lfp_band'].timestamps)
                            lfp_time_df = pd.DataFrame(lfp_timestamps)
                            lfp_time_df_1 = lfp_time_df.set_index(0)

                            # need to add smooth speed from position_df to lfp timestamps
                            merged_df = pd.merge_asof(lfp_time_df_1,spyglass_position_df, 
                                                    left_on = 0, right_on = 'time',
                                                    tolerance = 0.002,direction='nearest')
                            print(merged_df[merged_df['speed_smooth']>4].shape)

                            # try to replace find_nearest with digitize - can pass in all values at once
                            inner_loop_count = 0
                            for i in np.arange(1000):
                                index_list = np.digitize(center_well_timebins_adj[i,:],lfp_timestamps)
                                # find time diff and remove if greater than 2msec
                                try:
                                    close_match = np.where(np.abs(lfp_timestamps[index_list]-
                                                                center_well_timebins_adj[i,:])<0.002)
                                    random_df = merged_df.iloc[index_list[close_match]]
                                    mvt_random = random_df[random_df['speed_smooth']>4]
                                    if inner_loop_count < 5:
                                        print(mvt_random.shape)

                                    # get theta phase for these times
                                    column_name = 'electrode '+str(ref_elec_list[1])
                                    if inner_loop_count < 5:
                                        print(column_name)
                                    random_phase = theta_phase_df.loc[mvt_random['key_0'].values][column_name].values
                                except IndexError as e:
                                    print('missing data',key['nwb_file_name'],pos_name)
                                    random_phase = np.zeros((5,))
                                    print(random_phase)

                                # need to pad end of this string of numbers with NaN
                                padded = np.pad(random_phase,(0,1000),'constant',constant_values=(np.nan,np.nan))
                                theta_phase_random_array[i,:] = padded[0:1000]
                                inner_loop_count += 1

                            # need to look up the theta phase, dont just save the timestamp

        key['target_arm'] = 0
        key["theta_phase_random_array"] = theta_phase_random_array

        self.insert1(key)
