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
from spyglass.spikesorting import Waveforms
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

schema = dj.schema("mcoulter_realtime_lfp_spectogram")


# parameters to calculate place field
@schema
class LFPSpectogramParameters(dj.Manual):
    definition = """
    lfp_spect_param_name : varchar(500)
    ---
    lfp_spect_parameters : blob
    """

    # can include description above
    # description : varchar(500)

    def insert_default(self):
        lfp_spect_parameters = {}
        lfp_spect_parameters["well_distance_max"] = 17
        lfp_spect_parameters["center_well_pos"] = [449, 330]
        lfp_spect_parameters["arm_posterior_fraction"] = 0.4
        lfp_spect_parameters["center_posterior_fraction"] = 0.6
        lfp_spect_parameters['smoothing'] = 0
		# this is 100 msec
        lfp_spect_parameters['rip_prox'] = 3000
        lfp_spect_parameters['file_directory'] = 'placeholder'
        lfp_spect_parameters['min_reward_count'] = 67
        self.insert1(
            ["default", lfp_spect_parameters], skip_duplicates=True
        )


# select parameters
@schema
class LFPSpectogramSelection(dj.Manual):
    definition = """
    -> RealtimeFilename
    -> LFPSpectogramParameters
    ---
    """


# this is the computed table - basically it has the results of the analysis
# NOTE: for spectogram plotting, use longblob
@schema
class LFPSpectogram(dj.Computed):
    definition = """
    -> LFPSpectogramSelection
    ---
    all_arm1_times : double
    all_arm2_times : double
    all_random_times : double
    arm1_times_mvt : mediumblob
    arm1_times_rip : mediumblob
    arm1_times_still : mediumblob
    arm2_times_mvt : mediumblob
    arm2_times_rip : mediumblob
    arm2_times_still : mediumblob
    random_times_mvt : mediumblob
    random_times_rip : mediumblob
    random_times_still : mediumblob
    reward_times_mvt : blob
    reward_times_rip : blob
    reward_times_still : blob
    """

    def make(self, key):
        print(f"getting LFP times for: {key}")

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

        def remove_intersection(list1, list2):
            non_intersecting_list1 = []
            non_intersecting_list2 = []

            for start1, end1 in list1:
                intervals_to_add = [(start1, end1)]
                for start2, end2 in list2:
                    new_intervals = []
                    for a, b in intervals_to_add:
                        if b <= start2 or a >= end2:
                            new_intervals.append((a, b))
                            continue  # No overlap
                        if start2 > a:
                            new_intervals.append((a, start2))
                        if end2 < b:
                            new_intervals.append((end2, b))
                    intervals_to_add = new_intervals
                non_intersecting_list1.extend(
                    filter(lambda x: x[0] < x[1], intervals_to_add)
                )

            for start2, end2 in list2:
                intervals_to_add = [(start2, end2)]
                for start1, end1 in list1:
                    new_intervals = []
                    for a, b in intervals_to_add:
                        if b <= start1 or a >= end1:
                            new_intervals.append((a, b))
                            continue  # No overlap
                        if start1 > a:
                            new_intervals.append((a, start1))
                        if end1 < b:
                            new_intervals.append((end1, b))
                    intervals_to_add = new_intervals
                non_intersecting_list2.extend(
                    filter(lambda x: x[0] < x[1], intervals_to_add)
                )
            return non_intersecting_list1, non_intersecting_list2

        # StateScriptFile: get the time slice for taskState1
        # this should all come from the key
        # nwb_file_name = 'ginny20211101_.nwb'

        # need to check interval_list_name and nwb_file_name and include all possibilites here

        # insert code here

        lfp_posterior_sum_parameters = (
            LFPSpectogramParameters
            & {
                "lfp_spect_param_name": key[
                    "lfp_spect_param_name"
                ]
            }
        ).fetch()[0][1]

        post_thresh = lfp_posterior_sum_parameters["arm_posterior_fraction"]
        well_distance_max = lfp_posterior_sum_parameters["well_distance_max"]
        # smoothing: yes/no
        # time overlap with ripples
        smoothing_on = lfp_posterior_sum_parameters['smoothing']
        ripple_prox = lfp_posterior_sum_parameters['rip_prox']
        min_reward_count = lfp_posterior_sum_parameters['min_reward_count']
        speed_thresh = lfp_posterior_sum_parameters['speed_thresh']

        # set up variables
        arm1_end_timebins = np.zeros((1,))
        arm2_end_timebins = np.zeros((1,))
        center_well_timebins_sub = np.zeros((1,))
        arm1_end_overlap = []
        arm2_end_overlap = []
        reward_overlap = []
        arm1_times_mvt = []
        arm1_times_rip = []
        arm1_times_still = []
        arm2_times_mvt = []
        arm2_times_rip = []
        arm2_times_still = []
        random_times_mvt = []
        random_times_rip = []
        random_times_still = []

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
        if stim_message_diff_reward_2.shape[0] > min_reward_count:

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

            # add binary column for smoothed arm ends, then get intervals
            task2_decode_center['group_arm1'] = (task2_decode_center['arm1_end_smooth'] != 
                                                    task2_decode_center['arm1_end_smooth'].shift()).cumsum()
            arm1_intervals = (task2_decode_center[task2_decode_center['arm1_end_smooth'] == 1].
                                groupby('group_arm1').apply(lambda x: (x.index[0], x.index[-1])).tolist())
            task2_decode_center['group_arm2'] = (task2_decode_center['arm2_end_smooth'] != 
                                                    task2_decode_center['arm2_end_smooth'].shift()).cumsum()
            arm2_intervals = (task2_decode_center[task2_decode_center['arm2_end_smooth'] == 1].
                                groupby('group_arm2').apply(lambda x: (x.index[0], x.index[-1])).tolist())
            print('arm1 end intervals',len(arm1_intervals),'arm2 end intervals',len(arm2_intervals))
            # get timestamp for center bin
            arm1_interval_mid_list = []
            arm2_interval_mid_list = []
            for event in arm1_intervals:
                try:
                    midpoint_event = (event[1] - round((event[1]-event[0])/2))
                    mid_timestamp = task2_decode_center.loc[[midpoint_event]]['bin_timestamp'].values[0]
                    arm1_interval_mid_list.append(mid_timestamp)
                except KeyError as e:
                    print('event did not match')
            for event in arm2_intervals:
                try:
                    midpoint_event = (event[1] - round((event[1]-event[0])/2))
                    mid_timestamp = task2_decode_center.loc[[midpoint_event]]['bin_timestamp'].values[0]
                    arm2_interval_mid_list.append(mid_timestamp)
                except KeyError as e:
                    print('event did not match')
            arm1_smooth_midpoint = np.array(arm1_interval_mid_list)
            arm2_smooth_midpoint = np.array(arm2_interval_mid_list)

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

            # number of bins with >0.4 of posterior in arm end
            # if (
            #    task2_decode_center.shape[0] > 0
            #    and key["nwb_file_name"] != "molly20220322_.nwb"
            #    and key["nwb_file_name"] != "molly20220319_.nwb"
            #    and (
            #        key["nwb_file_name"] != "molly20220329_.nwb"
            #        and key["interval_list_name"] != "02_r1"
            #    )
            #    and (
            #        key["nwb_file_name"] != "molly20220324_.nwb"
            #        and key["interval_list_name"] != "02_r1"
            #    )
            #    and key["nwb_file_name"] != "arthur20220318_.nwb"
            # ):
            # temp for molly
            # if key["interval_list_name"] != "02_r1":

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

                #center_well_timebins = task2_decode_center[
                #    task2_decode_center["center_decode"] > 0.6
                #]["bin_timestamp"].values
                #center_well_timebins_sub = np.random.choice(
                #    center_well_timebins, average_arm_end_bins
                #)
                #print(
                #    "subsample center bins",
                #    average_arm_end_bins,
                #    center_well_timebins_sub.shape,
                #)

                # 1000 random bins from task2
                task2_all = decoder_data_center_well[decoder_data_center_well['taskState']==2]
                center_well_timebins = task2_all["bin_timestamp"].values
                center_well_timebins_sub = np.random.choice(center_well_timebins, 5000)

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

                pos_merge_id = PositionOutput.merge_get_part(
                    restriction=key_2
                ).fetch("merge_id")[0]

                print("getting ripple and offset", key["nwb_file_name"], pos_name)
                print(pos_merge_id)
                ripple_times = (
                    sgrip.RippleTimesV1()
                    & {"pos_merge_id": pos_merge_id}
                    & {'target_interval_list_name LIKE "%30ms%"'}
                ).fetch1_dataframe()
                print("number of rips", ripple_times.shape)
                # pos_name = PositionOutput.merge_get_part({'merge_id':pos_merge_id}
                #                                        ).fetch('interval_list_name')[filename_counter]

                # need to convert realtime timestamps to spyglass
                offset = (
                    StatescriptReward()
                    & {"nwb_file_name": key["nwb_file_name"]}
                    & {"interval_list_name": pos_name}
                ).fetch("offset")

                ripple_times["adjusted_start"] = (
                    ripple_times["start_time"] - offset
                ) * 30000
                ripple_times["adjusted_end"] = (
                    ripple_times["end_time"] - offset
                ) * 30000

                # make a list of center well times
                realtime_content_time_array = np.array(
                    task2_decode_center["bin_timestamp"].values
                )
                diff_index = np.where(np.diff(realtime_content_time_array) > 180)[0]

                interval_list = []
                if diff_index.shape[0] > 0:
                    # first entry
                    interval_list.append(
                        [
                            realtime_content_time_array[0] / 30000 + offset[0],
                            realtime_content_time_array[diff_index[0]] / 30000
                            + offset[0],
                        ]
                    )
                    if diff_index.shape[0] > 1:
                        for i in np.arange(diff_index.shape[0] - 1):
                            interval_list.append(
                                [
                                    realtime_content_time_array[diff_index[i] + 1]
                                    / 30000
                                    + offset[0],
                                    realtime_content_time_array[diff_index[i + 1]]
                                    / 30000
                                    + offset[0],
                                ]
                            )
                        # last interval
                        interval_list.append(
                            [
                                realtime_content_time_array[diff_index[i + 1] + 1]
                                / 30000
                                + offset[0],
                                realtime_content_time_array[-1] / 30000 + offset[0],
                            ]
                        )
                    # only 1 time break
                    else:
                        interval_list.append(
                            [
                                realtime_content_time_array[diff_index[0] + 1]
                                / 30000
                                + offset[0],
                                realtime_content_time_array[-1] / 30000 + offset[0],
                            ]
                        )
                else:
                    print("only 1 center interval")
                    interval_list.append(
                        [
                            realtime_content_time_array[0] / 30000 + offset[0],
                            realtime_content_time_array[-1] / 30000 + offset[0],
                        ]
                    )
                # print(interval_list)

                # make list of ripple intervals
                ripple_times
                ripple_time_list = []
                for ripple in np.arange(ripple_times.shape[0]):
                    ripple_time_list.append(
                        [
                            ripple_times["start_time"].values[ripple],
                            ripple_times["end_time"].values[ripple],
                        ]
                    )

                # center well times during ripples
                ripple_center_times = interval_list_intersect(
                    np.array(interval_list), np.array(ripple_time_list)
                )
                if len(ripple_center_times) > 0:
                    center_ripple_df = pd.DataFrame(
                        np.concatenate(
                            [
                                np.expand_dims(
                                    np.array(ripple_center_times)[:, 0], axis=1
                                ),
                                np.expand_dims(
                                    np.array(ripple_center_times)[:, 1], axis=1
                                ),
                            ],
                            axis=1,
                        ),
                        columns=["start_time", "end_time"],
                    )
                    center_ripple_df["adjusted_start"] = (
                        center_ripple_df["start_time"] - offset
                    ) * 30000
                    center_ripple_df["adjusted_end"] = (
                        center_ripple_df["end_time"] - offset
                    ) * 30000

                    # center well times - no ripple
                    (
                        non_intersecting_list1,
                        non_intersecting_list2,
                    ) = remove_intersection(interval_list, ripple_time_list)
                    center_non_ripple_df = pd.DataFrame(
                        np.concatenate(
                            [
                                np.expand_dims(
                                    np.array(non_intersecting_list1)[:, 0], axis=1
                                ),
                                np.expand_dims(
                                    np.array(non_intersecting_list1)[:, 1], axis=1
                                ),
                            ],
                            axis=1,
                        ),
                        columns=["start_time", "end_time"],
                    )
                    center_non_ripple_df["adjusted_start"] = (
                        center_non_ripple_df["start_time"] - offset
                    ) * 30000
                    center_non_ripple_df["adjusted_end"] = (
                        center_non_ripple_df["end_time"] - offset
                    ) * 30000

                    # calculated ripple consensus trace
                    key_3 = (
                        sgrip.RippleTimesV1()
                        & {"nwb_file_name": key["nwb_file_name"]}
                        & {"pos_merge_id": pos_merge_id}
                    ).fetch("KEY")[0]
                    ripple_params = (
                        sgrip.RippleParameters
                        & {"ripple_param_name": key_3["ripple_param_name"]}
                    ).fetch1("ripple_param_dict")
                    ripple_detection_params = ripple_params[
                        "ripple_detection_params"
                    ]

                    (
                        speed,
                        interval_ripple_lfps,
                        sampling_frequency,
                    ) = sgrip.RippleTimesV1.get_ripple_lfps_and_position_info(
                        key_3, key["nwb_file_name"], pos_name
                    )
                    lfp_timestamps = np.asarray(interval_ripple_lfps.index)

                    #consensus_trace = get_Kay_ripple_consensus_trace(
                    #    np.asarray(interval_ripple_lfps),
                    #    sampling_frequency,
                    #    smoothing_sigma=0.004,
                    #)
                    #consensus_trace_zscore = zscore(
                    #    consensus_trace, nan_policy="omit"
                    #)

                    # need to make a loop here for task2_center, with ripple, without ripples
                    # want to figure out which timebins overlap with ripples
                    arm1_rip_overlap_times = []
                    arm2_rip_overlap_times = []
                    random_rip_overlap_times = []
                    # try just rips_only
                    #for overlap_comp in [
                    #    "rips_only",
                    #    "rips_center",
                    #    "no_rips_center",
                    #]:
                    for overlap_comp in ['rips_only']:
                        # for overlap_comp in ['rips_only','rips_center',]:
                        # print('LOOP',overlap_comp)
                        for location in [1, 2, 4]:
                            if location == 1:
                                if smoothing_on == 1:
                                    target_timebins = arm1_smooth_midpoint.copy()
                                else:
                                    target_timebins = arm1_end_timebins.copy()
                            elif location == 2:
                                if smoothing_on == 1:
                                    target_timebins = arm2_smooth_midpoint.copy()
                                else:
                                    target_timebins = arm2_end_timebins.copy()
                            elif location == 3:
                                target_timebins = stim_message_diff_reward_2[
                                    "bin_timestamp"
                                ].values
                            elif location == 4:
                                target_timebins = center_well_timebins_sub.copy()

                            if location == 3:
                                a = target_timebins - 4 * 180
                                b = target_timebins + 180

                            else:
                                if overlap_comp == "rips_only":
                                    a = target_timebins
                                    b = target_timebins + 180
                                    # if smoothing
                                    if smoothing_on == 1:
                                        a = target_timebins - ripple_prox
                                        b = target_timebins + ripple_prox
                                else:
                                    # print('rips center')
                                    # central 1 sec: 75 - 105
                                    a = target_timebins + 75
                                    b = target_timebins + 105

                            if overlap_comp == "rips_only":
                                c = ripple_times["adjusted_start"]
                                d = ripple_times["adjusted_end"]
                            elif overlap_comp == "rips_center":
                                # print('rips center C and D')
                                c = center_ripple_df["adjusted_start"]
                                d = center_ripple_df["adjusted_end"]
                            elif overlap_comp == "no_rips_center":
                                c = center_non_ripple_df["adjusted_start"]
                                d = center_non_ripple_df["adjusted_end"]

                            target_arm_end_df = pd.DataFrame(
                                {
                                    "start": a,
                                    "end": b,
                                }
                            )
                            ripples_merge_df = pd.DataFrame(
                                {
                                    "start": c,
                                    "end": d,
                                }
                            )

                            arm_end_intervals = pd.IntervalIndex.from_arrays(
                                target_arm_end_df.start,
                                target_arm_end_df.end,
                                closed="both",
                            )
                            ripples_intervals = pd.IntervalIndex.from_arrays(
                                ripples_merge_df.start,
                                ripples_merge_df.end,
                                closed="both",
                            )
                            # print(ripples_intervals[0:2])

                            overlap_list = []

                            for interval in arm_end_intervals:
                                # print(np.sum(list_1.overlaps(interval)))
                                overlap_list.append(
                                    np.sum(ripples_intervals.overlaps(interval))
                                )
                                # can we add the time to a ripple overlap list?
                                if np.sum(ripples_intervals.overlaps(interval)) == 1 and location == 1 and overlap_comp == 'rips_only':
                                    #print('arm 1 interval',interval)
                                    arm1_rip_overlap_times.append(interval.left)
                                elif np.sum(ripples_intervals.overlaps(interval)) > 1 and location == 1 and overlap_comp == 'rips_only':
                                    print('overlaps 2 or more riplpes, arm1')

                                if np.sum(ripples_intervals.overlaps(interval)) == 1 and location == 2 and overlap_comp == 'rips_only':
                                    #print('arm 2 interval',interval)
                                    arm2_rip_overlap_times.append(interval.left)
                                elif np.sum(ripples_intervals.overlaps(interval)) > 1 and location == 2 and overlap_comp == 'rips_only':
                                    print('overlaps 2 or more riplpes, arm2')

                                if np.sum(ripples_intervals.overlaps(interval)) == 1 and location == 4 and overlap_comp == 'rips_only':
                                    #print('arm 2 interval',interval)
                                    random_rip_overlap_times.append(interval.left)
                                elif np.sum(ripples_intervals.overlaps(interval)) > 1 and location == 4 and overlap_comp == 'rips_only':
                                    print('overlaps 2 or more riplpes, random')                                                                            
                            overlap_array = np.array(overlap_list)
                            # print('overlap array non-zero',overlap_comp,np.nonzero(overlap_array)[0].shape[0])

                    #print(
                    #    "center rips times",
                    #    summed_time_1 / 166.7,
                    #    arm1_end_overlap_rip[-1],
                    #    arm2_end_overlap_rip[-1],
                    #)
                    #print(
                    #    "center non rips times",
                    #    summed_time_2 / 166.7,
                    #    arm1_end_overlap_norip[-1],
                    #    arm2_end_overlap_norip[-1],
                    #)
                    print("overlapping fraction")
                    print(key["nwb_file_name"], pos_name)
                    # print(np.nonzero(overlap_array)[0].shape,overlap_array.shape)

                    # find speed for each arm end bin
                    # also find ripple conensus z-score
                    # add condtional for smoothed decode
                    if smoothing_on == 1:
                        process_timebins = arm1_smooth_midpoint.copy()
                    else:
                        process_timebins = arm1_end_timebins.copy()

                    for time in process_timebins:

                        adjusted_timestamp = time / 30000 + offset
                        # print(adjusted_timestamp[0])
                        new_time = find_nearest(
                            spyglass_position_df.index, adjusted_timestamp
                        )
                        new_speed = spyglass_position_df.loc[[new_time]][
                            "speed_smooth"
                        ].values[0]
                        #all_speeds_arm1.append(new_speed)

                        # if not using this, then comment out
                        # NOTE: should only use if this is a close match (min_diff < 0.02)
                        rip_time, placeholder, min_diff = find_nearest_1(
                            lfp_timestamps, adjusted_timestamp
                        )
                        # add time to list for either mvt or still
                        if time in arm1_rip_overlap_times:
                            arm1_times_rip.append(new_time)                       
                        elif new_speed > speed_thresh and min_diff < 0.02:
                            arm1_times_mvt.append(new_time)
                        elif new_speed < speed_thresh and min_diff < 0.02:
                            arm1_times_still.append(new_time)

                    print(
                        "done with arm 1",
                        arm1_end_timebins.shape,
                    )

                    # add condtional for smoothed decode
                    if smoothing_on == 1:
                        process_timebins = arm2_smooth_midpoint.copy()
                    else:
                        process_timebins = arm2_end_timebins.copy()

                    for time in process_timebins:
                        adjusted_timestamp = time / 30000 + offset
                        # print(adjusted_timestamp[0])
                        new_time = find_nearest(
                            spyglass_position_df.index, adjusted_timestamp
                        )
                        new_speed = spyglass_position_df.loc[[new_time]][
                            "speed_smooth"
                        ].values[0]
                        #all_speeds_arm2.append(new_speed)

                        # if not using this, then comment out
                        rip_time, placeholder, min_diff = find_nearest_1(
                            lfp_timestamps, adjusted_timestamp
                        )
                        # add time to list for either mvt or still
                        if time in arm2_rip_overlap_times:
                            arm2_times_rip.append(new_time)  
                        elif new_speed > speed_thresh and min_diff < 0.02:
                            arm2_times_mvt.append(new_time)
                        elif new_speed < speed_thresh and min_diff < 0.02:
                            arm2_times_still.append(new_time)

                    print(
                        "arm 2 timebins",
                        arm2_end_timebins.shape,
                    )

                    for time in target_timebins:
                        adjusted_timestamp = time / 30000 + offset
                        # print(adjusted_timestamp[0])
                        new_time = find_nearest(
                            spyglass_position_df.index, adjusted_timestamp
                        )
                        new_speed = spyglass_position_df.loc[[new_time]][
                            "speed_smooth"
                        ].values[0]
                        #all_speeds_target.append(new_speed)

						# if not using this, then comment out
                        #rip_time, placeholder, min_diff = find_nearest_1(
                        #    lfp_timestamps, adjusted_timestamp
                        #)
                        # add time to list for either mvt or still

                    for time in center_well_timebins_sub:
                        adjusted_timestamp = time / 30000 + offset
                        # print(adjusted_timestamp[0])
                        new_time = find_nearest(
                            spyglass_position_df.index, adjusted_timestamp
                        )
                        new_speed = spyglass_position_df.loc[[new_time]][
                            "speed_smooth"
                        ].values[0]
                        #all_speeds_center.append(new_speed)

                        # if not using this, then comment out
                        rip_time, placeholder, min_diff = find_nearest_1(
                            lfp_timestamps, adjusted_timestamp
                        )
                        # add time to list for either mvt or still
                        if time in random_rip_overlap_times:
                            random_times_rip.append(new_time)  
                        elif new_speed > speed_thresh and min_diff < 0.02:
                            random_times_mvt.append(new_time)
                        elif new_speed < speed_thresh and min_diff < 0.02:
                            random_times_still.append(new_time)

                    # this can check that the two intervals are lined up
                    # plt.figure()
                    # plt.scatter(target_arm_end_df['start'],np.repeat(1,target_arm_end_df.shape[0]))
                    # plt.scatter(ripples_df['start'],np.repeat(2,ripples_df.shape[0]))

        # summary stats
        print('arm1. mvt: ',len(arm1_times_mvt),'rip:',len(arm1_times_rip),'still:',len(arm1_times_still))
        print('arm2. mvt: ',len(arm2_times_mvt),'rip:',len(arm2_times_rip),'still:',len(arm2_times_still))
        print('random. mvt: ',len(random_times_mvt),'rip:',len(random_times_rip),'still:',len(random_times_still))

        if smoothing_on == 1:
            key['all_arm1_times'] = arm1_smooth_midpoint.shape[0]
            key['all_arm2_times'] = arm2_smooth_midpoint.shape[0]
            key['all_random_times'] = center_well_timebins_sub.shape[0]
        else:
            key['all_arm1_times'] = arm1_end_timebins.shape[0]
            key['all_arm2_times'] = arm2_end_timebins.shape[0]
            key['all_random_times'] = center_well_timebins_sub.shape[0]
        key["arm1_times_mvt"] = arm1_times_mvt
        key["arm1_times_rip"] = arm1_times_rip
        key["arm1_times_still"] = arm1_times_still
        key["arm2_times_mvt"] = arm2_times_mvt
        key["arm2_times_rip"] = arm2_times_rip
        key["arm2_times_still"] = arm2_times_still
        key["random_times_mvt"] = random_times_mvt
        key["random_times_rip"] = random_times_rip
        key["random_times_still"] = random_times_still
        key["reward_times_mvt"] = []
        key["reward_times_rip"] = []
        key["reward_times_still"] = []        
  
        self.insert1(key)
