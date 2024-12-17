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

        def find_intervals_exceeding_threshold(arr, x):
            # Create a binary mask where values exceed the threshold
            binary_array = arr > x

            # Find start and end of contiguous True segments
            def find_intervals(binary_array):
                intervals = []
                start = None

                for i, value in enumerate(binary_array):
                    if value and start is None:
                        start = i
                    elif not value and start is not None:
                        intervals.append((start, i - 1))
                        start = None

                if start is not None:
                    intervals.append((start, len(binary_array) - 1))

                return intervals

            # Get the intervals
            intervals = find_intervals(binary_array)

            # Return intervals with their corresponding values
            result = []
            for start_idx, end_idx in intervals:
                span_values = arr[start_idx:end_idx + 1]
                span_indices = np.arange(start_idx, end_idx + 1)
                result.append({
                    "indices": span_indices,
                    "values": span_values
                })

            return result

        def _intersection(interval1, interval2):
            """Takes the (set-theoretic) intersection of two intervals"""
            start = max(interval1[0], interval2[0])
            end = min(interval1[1], interval2[1])
            # i think this needs to change for pandas interval
            #start = max(interval2[0], interval1.left)
            #end = min(interval2[1], interval1.right)
            intersection = np.array([start, end]) if end > start else None
            #return intersection
            if intersection is not None:
                overlap = interval2.copy()
            else:
                overlap = None

            return overlap

        def interval_list_intersect_1(interval_list1, interval_list2, min_length=0):
            # then do pairwise comparison and collect intersections
            overlap = [
                _intersection(interval2, interval1)
                for interval2 in interval_list2
                for interval1 in interval_list1
                if _intersection(interval2, interval1) is not None
            ]

            return overlap

        def group_intervals(arr, max_diff):
            # List to store the intervals
            intervals = []
            # Start and end index of the current interval
            start = 0
            # Iterate through the array to form intervals
            for i in range(1, len(arr)):
                if abs(arr[i] - arr[i - 1]) >= max_diff:
                    # If the difference between adjacent values is >= max_diff,
                    # close the current interval and start a new one
                    intervals.append({
                        'indices': np.arange(start, i),
                        'values': arr[start:i]
                    })
                    start = i
            # Append the last interval
            intervals.append({
                'indices': np.arange(start, len(arr)),
                'values': arr[start:]
            })
            return intervals

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

        # target dict for all days
        ron_all_dict = {'ron20210811_.nwb':2,'ron20210812_.nwb':2,
                            'ron20210813_.nwb':1,'ron20210814_.nwb':1,
                        'ron20210816_.nwb':2,'ron20210817_.nwb':2,'ron20210818_.nwb':2,
                        'ron20210819_.nwb':1,'ron20210820_.nwb':1,'ron20210821_.nwb':1,
                        'ron20210822_.nwb':2,'ron20210823_.nwb':2,'ron20210824_.nwb':2,
                        'ron20210825_.nwb':1,'ron20210826_.nwb':1,'ron20210827_.nwb':1,
                        'ron20210828_.nwb':2,'ron20210829_.nwb':2,'ron20210830_.nwb':2,
                        'ron20210831_.nwb':1,'ron20210901_.nwb':1,'ron20210902_.nwb':1}

        tonks_all_dict = {'tonks20211023_.nwb': 1,'tonks20211024_.nwb': 1,
                            'tonks20211025_.nwb': 2,'tonks20211026_.nwb': 2,
        'tonks20211027_.nwb': 1,'tonks20211028_.nwb': 1,'tonks20211029_.nwb': 1,
        'tonks20211030_.nwb': 2,'tonks20211031_.nwb': 2,'tonks20211101_.nwb': 2,
        'tonks20211102_.nwb': 1,'tonks20211103_.nwb': 1,'tonks20211104_.nwb': 1,
        'tonks20211105_.nwb': 2,'tonks20211106_.nwb': 2,'tonks20211107_.nwb': 2,
        'tonks20211108_.nwb': 1,'tonks20211109_.nwb': 1,'tonks20211110_.nwb': 1,
        'tonks20211111_.nwb': 2,'tonks20211112_.nwb': 2,}

        ginny_all_dict = {'ginny20211023_.nwb': 1,'ginny20211024_.nwb': 1,
                            'ginny20211025_.nwb': 2,'ginny20211026_.nwb': 2,
        'ginny20211027_.nwb': 1,'ginny20211028_.nwb': 1,'ginny20211029_.nwb': 1,
        'ginny20211030_.nwb': 2,'ginny20211031_.nwb': 2,'ginny20211101_.nwb': 2,
        'ginny20211102_.nwb': 1,'ginny20211103_.nwb': 1,'ginny20211104_.nwb': 1,
        'ginny20211105_.nwb': 2,'ginny20211106_.nwb': 2,'ginny20211107_.nwb': 2,
        'ginny20211108_.nwb': 1,'ginny20211109_.nwb': 1,'ginny20211110_.nwb': 1,
        'ginny20211111_.nwb': 2,'ginny20211112_.nwb': 2,'ginny20211113_.nwb': 2,}

        molly_all_dict = {'molly20220314_.nwb': 1,'molly20220315_.nwb': 1,
                            'molly20220316_.nwb': 2,'molly20220317_.nwb': 2,
        'molly20220318_.nwb': 1,'molly20220319_.nwb': 1,'molly20220320_.nwb': 1,
        'molly20220321_.nwb': 2,'molly20220322_.nwb': 2,'molly20220323_.nwb': 2,
        'molly20220324_.nwb': 1,'molly20220325_.nwb': 1,'molly20220326_.nwb': 1,
        'molly20220327_.nwb': 2,'molly20220328_.nwb': 2,'molly20220329_.nwb': 2,
        'molly20220330_.nwb': 1,'molly20220331_.nwb': 1,'molly20220401_.nwb': 1,
        'molly20220402_.nwb': 2,'molly20220403_.nwb': 2,'molly20220404_.nwb': 2,}

        arthur_all_dict = {'arthur20220314_.nwb': 1,'arthur20220315_.nwb': 1,
                                'arthur20220316_.nwb': 2,'arthur20220317_.nwb': 2,
        'arthur20220318_.nwb': 1,'arthur20220319_.nwb': 1,'arthur20220320_.nwb': 1,
        'arthur20220321_.nwb': 2,'arthur20220322_.nwb': 2,'arthur20220323_.nwb': 2,
        'arthur20220324_.nwb': 1,'arthur20220325_.nwb': 1,'arthur20220326_.nwb': 1,
        'arthur20220327_.nwb': 2,'arthur20220328_.nwb': 2,'arthur20220329_.nwb': 2,
        'arthur20220330_.nwb': 1,'arthur20220331_.nwb': 1,'arthur20220401_.nwb': 1,
        'arthur20220402_.nwb': 2,'arthur20220403_.nwb': 2,'arthur20220404_.nwb': 2,}

        pippin_all_dict = {'pippin20210518_.nwb':1,'pippin20210519_.nwb':1,
                                'pippin20210520_.nwb':2,'pippin20210521_.nwb':2,
        'pippin20210523_.nwb':1,'pippin20210524_.nwb':1,'pippin20210525_.nwb':1,
        'pippin20210526_.nwb':2,'pippin20210527_.nwb':2,'pippin20210528_.nwb':2}



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

        print('smoothing:',smoothing_on)

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
        unique_rip_arm1 = []
        unique_rip_arm2 = []

        rat_name = key["nwb_file_name"].split("2")[0]

        # how to get realtime filename - not needed
        if rat_name == 'ron':
           path_add = "/cumulus/mcoulter/george/realtime_rec_merged/"
           target_arm = ron_all_dict[key['nwb_file_name']]
        elif rat_name == 'tonks':
           path_add = "/cumulus/mcoulter/tonks/realtime_merged/"
           target_arm = tonks_all_dict[key['nwb_file_name']]
        elif rat_name == 'arthur':
           path_add = "/cumulus/mcoulter/molly/realtime_rec_merged/"
           target_arm = arthur_all_dict[key['nwb_file_name']]
        elif rat_name == 'molly':
           path_add = "/cumulus/mcoulter/molly/realtime_rec_merged/"
           target_arm = molly_all_dict[key['nwb_file_name']]
        elif rat_name == 'ginny':
           path_add = "/cumulus/mcoulter/tonks/realtime_merged/"
           target_arm = ginny_all_dict[key['nwb_file_name']]
        elif rat_name == 'pippin':
           path_add = "/cumulus/mcoulter/pippin/realtime_rec_merged_end_all/"
           target_arm = pippin_all_dict[key['nwb_file_name']]

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
        if key["nwb_file_name"] == "pippin20210523_.nwb" and key["interval_list_name"] == '04_r2':
            pos_name = 'pos 3 valid times'
        elif key["nwb_file_name"] == "pippin20210523_.nwb" and key["interval_list_name"] == '06_r3':
            pos_name = 'pos 5 valid times'
        elif key["nwb_file_name"] == "pippin20210524_.nwb" and key["interval_list_name"] == '04_r2':
            pos_name = 'pos 3 valid times'
        elif key["nwb_file_name"] == "pippin20210524_.nwb" and key["interval_list_name"] == '06_r3':
            pos_name = 'pos 5 valid times'
        elif key["interval_list_name"] == realtime_interval_list[0]:
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


        rip_times_shuf_list = []
        still_times_shuf_list = []
        mvt_times_shuf_list = []
        target_ripple = []
        target_non_rip = []
        target_mvt = []
        target_still = []
        target_rip_remove = []

        # only high reward sessions
        if stim_message_diff_reward_2.shape[0] > min_reward_count:

            # fred, pippin, percy
            if rat_name == "pippin":
                center_well_pos = [634, 648]
            else:
                # geroge, ron, ginny, tonks
                center_well_pos = [449, 330]

            decoder_data_center_well = decoder_data.copy()

            decoder_timestamps = decoder_data['bin_timestamp'].values

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

            if smoothing_on == 1:
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
            else:
                arm1_smooth_midpoint = []
                arm2_smooth_midpoint = []

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

                if key["lfp_spect_param_name"] == 'speed_4_random':
                    center_well_timebins = task2_decode_center['bin_timestamp'].values
                    center_well_timebins_sub = np.random.choice(center_well_timebins, average_arm_end_bins)
                else:
                    # 1000 random bins from task2
                    task2_all = decoder_data_center_well[decoder_data_center_well['taskState']==2]
                    center_well_timebins = task2_all["bin_timestamp"].values
                    center_well_timebins_sub = np.random.choice(center_well_timebins, 5000)

                # should update this to use task2_decode_center and take a number of samples that is the average of arm 1 + 2

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
                if key["nwb_file_name"] == 'arthur20220314_.nwb':
                    sort_intervals = ['r1_r2','r2_r3']
                else:
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
                if key["nwb_file_name"] == "pippin20210523_.nwb" and pos_name == 'pos 3 valid times':
                    sort_interval = sort_intervals[0]
                elif key["nwb_file_name"] == "pippin20210523_.nwb" and pos_name == 'pos 5 valid times':
                    sort_interval = sort_intervals[1]
                elif key["nwb_file_name"] == "pippin20210524_.nwb" and pos_name == 'pos 3 valid times':
                    sort_interval = sort_intervals[0]
                elif key["nwb_file_name"] == "pippin20210524_.nwb" and pos_name == 'pos 5 valid times':
                    sort_interval = sort_intervals[1]
                elif pos_name == all_pos[0]:
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
                    ) = sgrip.RippleTimesV1.get_ripple_lfps_and_position_info(key_3)
                    lfp_timestamps = np.asarray(interval_ripple_lfps.index)

                    consensus_trace = get_Kay_ripple_consensus_trace(
                        np.asarray(interval_ripple_lfps),
                        sampling_frequency,
                        smoothing_sigma=0.004,
                    )
                    consensus_trace_zscore = zscore(
                        consensus_trace, nan_policy="omit"
                    )

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

                        # put the 1000 shuffle loop here
                        if key["lfp_spect_param_name"] == 'random_shuffle':
                            shuf_count = 1000
                            print('random shuffle',shuf_count)
                        else:
                            shuf_count = 1
                        for i in np.arange(shuf_count):
                            if key["lfp_spect_param_name"] == 'random_shuffle':
                                center_well_timebins = task2_decode_center['bin_timestamp'].values
                                center_well_timebins_sub = np.random.choice(center_well_timebins, 1000)

                            #for location in [1, 2, 4]:
                            # 8-5-24: random only
                            for location in [1]:
                                if key["lfp_spect_param_name"] != 'whole_post_smooth':
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

                                # different way of finding remote decode and ripple overlap
                                elif key["lfp_spect_param_name"] == 'whole_post_smooth':
                                    # somehow want to match these up with other times - to check ripple and speed
                                    target_timebins = stim_message_diff_reward_2["bin_timestamp"].values
                                    target_adj = (target_timebins/30000) + offset

                                    event_length_list_arm1 = []
                                    event_length_list_arm2 = []
                                    smooth_posterior_whole = ndimage.gaussian_filter1d((decoder_data.iloc[:,27:68]),1,axis=0)
                                    #rewarded_representation_1[time_counter,:,:] = surround_posterior
                                    arm1_end_sum = np.sum(smooth_posterior_whole[:,20:25],axis=1)
                                    arm2_end_sum = np.sum(smooth_posterior_whole[:,36:],axis=1)

                                    # equivalent to 0.4 without smooth is 0.268 with smooth
                                    remote_intervals_arm1 = find_intervals_exceeding_threshold(arm1_end_sum, 0.268)
                                    remote_intervals_arm2 = find_intervals_exceeding_threshold(arm2_end_sum, 0.268)

                                    # remove intervals that go over a time gap - does nothing
                                    #for interval in remote_intervals_arm1:
                                    #    event_timestamps = decoder_timestamps[interval['indices']]
                                    #    if np.any(np.diff(event_timestamps) > 180) and event_timestamps.shape[0] > 1:
                                    #        print('event with gap',event_timestamps)
                                    #    else:
                                    #        event_length_list_arm1.append(interval['indices'].shape[0])

                                    #for interval in remote_intervals_arm2:
                                    #    event_timestamps = decoder_timestamps[interval['indices']]
                                    #    if np.any(np.diff(event_timestamps) > 180) and event_timestamps.shape[0] > 1:
                                    #        print('event with gap',event_timestamps)
                                    #    else:
                                    #        event_length_list_arm2.append(interval['indices'].shape[0])
                                    #print('arm1 times',len(event_length_list_arm1),
                                    #      'arm2 times',len(event_length_list_arm2))

                                    # arm 1
                                    # this seems to work for brain state - to make interval list
                                    #intervals = group_intervals(target_still, 1)
                                    #arm_end_list_intervals = []
                                    #for interval in intervals:
                                    #    arm_end_list_intervals.append([interval['values'][0],interval['values'][-1]])

                                    # use ripples intervals?

                                    # now need to make a list from realtime decoder
                                    # only keep event at center well
                                    posterior_interval_list = []
                                    center_well_timebins = task2_decode_center['bin_timestamp'].values

                                    for interval in remote_intervals_arm1:
                                        placeholder, placeholder2, time_diff = find_nearest_1(center_well_timebins,
                                                                  decoder_timestamps[interval['indices'][0]])
                                        if interval['indices'].shape[0] == 1 and time_diff <= 180:
                                            posterior_interval_list.append([((decoder_timestamps[interval['indices']]/30000)+offset)[0],
                                                                        ((decoder_timestamps[interval['indices']+1]/30000)+offset)[0]])
                                        elif time_diff <= 180:
                                            posterior_interval_list.append([((decoder_timestamps[interval['indices'][0]]/30000)+offset)[0],
                                                                        ((decoder_timestamps[interval['indices'][-1]+1]/30000)+offset)[0]])
                                        else:
                                            pass
                                    print('all arm1',len(remote_intervals_arm1),'center well',len(posterior_interval_list))

                                    # find posterior times that overlap with ripples
                                    rip_overlap_post = interval_list_intersect_1(np.array(posterior_interval_list),
                                                                            np.array(ripple_time_list))
                                    unique_rip_arm1 = np.unique(np.array(rip_overlap_post),axis=0)
                                    print('overlap with rips arm1',unique_rip_arm1.shape)

                                    # try to print ripple power for overlapping events
                                    arm1_ripple_power = []
                                    for single_rip in unique_rip_arm1:
                                        remote_start = find_nearest(lfp_timestamps,single_rip[0])
                                        remote_end = find_nearest(lfp_timestamps,single_rip[1])
                                        start_ind = np.where(lfp_timestamps==remote_start)[0][0]
                                        end_ind = np.where(lfp_timestamps==remote_end)[0][0]
                                        arm1_ripple_power.append(np.max(consensus_trace_zscore[start_ind-5:end_ind+5]))

                                        # match time to rewards - i think with will work with small offset 0.002 or 0.01
                                        target_time, placeholder, time_diff = find_nearest_1(target_adj,
                                                                                (single_rip[0]+single_rip[1])/2)
                                        if target_arm == 1 and time_diff <= 0.02:
                                            target_ripple.append(single_rip)
                                            target_rip_remove.append(target_time)
                                            # save this to table
                                        elif target_arm == 1:
                                            # check time diff
                                            #print('time diff to reward',time_diff)
                                            #target_non_rip.append(target_time)
                                            # check speed for these
                                            # NOTE: do the same for arm 2
                                            # may want to find target arm
                                            pass

                                    #print('ripple power arm 1',arm1_ripple_power[0:100])
                                    print('arm1. non ripple targets:',len(target_non_rip),
                                          'overlap with ripples',len(target_rip_remove))
                                    # remove ripple/target overlaps
                                    target_non_rip = np.setdiff1d(target_adj,np.array(target_rip_remove))

                                    index_list = []
                                    for single_overlap in unique_rip_arm1:
                                        index_list.append(np.where(single_overlap == posterior_interval_list))

                                    # remove those from list of all posterior times
                                    non_overlap_list = np.delete(np.array(posterior_interval_list),np.array(index_list)[:,0][:,0],axis=0)
                                    # now: take center of each interval and get speed
                                    #midpoint_list = []
                                    #for counter in np.arange(non_overlap_list.shape[0]):
                                    #    midpoint_list.append((non_overlap_list[counter,0]+non_overlap_list[counter,1])/2)
                                    #arm1_end_timebins = midpoint_list.copy()
                                    arm1_end_timebins = non_overlap_list.copy()

                                    # arm 2
                                    # this seems to work for brain state - to make interval list
                                    #intervals = group_intervals(target_still, 1)
                                    #arm_end_list_intervals = []
                                    #for interval in intervals:
                                    #    arm_end_list_intervals.append([interval['values'][0],interval['values'][-1]])

                                    # now need to make a list from realtime decoder
                                    posterior_interval_list = []
                                    for interval in remote_intervals_arm2:
                                        placeholder, placeholder2, time_diff = find_nearest_1(center_well_timebins,
                                                                  decoder_timestamps[interval['indices'][0]])
                                        if interval['indices'].shape[0] == 1 and time_diff <= 180:
                                            posterior_interval_list.append([((decoder_timestamps[interval['indices']]/30000)+offset)[0],
                                                                        ((decoder_timestamps[interval['indices']+1]/30000)+offset)[0]])
                                        elif time_diff <= 180:
                                            posterior_interval_list.append([((decoder_timestamps[interval['indices'][0]]/30000)+offset)[0],
                                                                        ((decoder_timestamps[interval['indices'][-1]+1]/30000)+offset)[0]])
                                        else:
                                            pass
                                    # find posterior times that overlap with ripples
                                    rip_overlap_post = interval_list_intersect_1(np.array(posterior_interval_list),
                                                                            np.array(ripple_time_list))
                                    unique_rip_arm2 = np.unique(np.array(rip_overlap_post),axis=0)
                                    print('all arm2',len(remote_intervals_arm2),'center well',len(posterior_interval_list))
                                    print('overlap with rips arm2',unique_rip_arm2.shape)
                                    # try to print ripple power for overlapping events
                                    arm2_ripple_power = []
                                    for single_rip in unique_rip_arm2:
                                        remote_start = find_nearest(lfp_timestamps,single_rip[0])
                                        remote_end = find_nearest(lfp_timestamps,single_rip[1])
                                        start_ind = np.where(lfp_timestamps==remote_start)[0][0]
                                        end_ind = np.where(lfp_timestamps==remote_end)[0][0]
                                        arm2_ripple_power.append(np.max(consensus_trace_zscore[start_ind-5:end_ind+5]))

                                        # match time to rewards - i think with will work with small offset 0.002 or 0.01
                                        target_time, placeholder, time_diff = find_nearest_1(target_adj,
                                                                                (single_rip[0]+single_rip[1])/2)
                                        if target_arm == 2 and time_diff <= 0.02:
                                            target_ripple.append(single_rip)
                                            target_rip_remove.append(target_time)
                                            # save this to table
                                        elif target_arm == 2:
                                            # check time diff
                                            #print('time diff to reward',time_diff)
                                            #target_non_rip.append(target_time)
                                            # check speed for these
                                            # NOTE: do the same for arm 2
                                            # may want to find target arm
                                            pass

                                    #print('ripple power arm 2',arm2_ripple_power[0:100])
                                    # remove targets with ripple from list
                                    target_non_rip = np.setdiff1d(target_adj,np.array(target_rip_remove))

                                    print('arm2. non ripple targets:',len(target_non_rip),
                                           'overlap with ripples',len(target_rip_remove))


                                    index_list = []
                                    for single_overlap in unique_rip_arm2:
                                        index_list.append(np.where(single_overlap == posterior_interval_list))
                                    # remove those from list of all posterior times
                                    non_overlap_list = np.delete(np.array(posterior_interval_list),np.array(index_list)[:,0][:,0],axis=0)
                                    # now: take center of each interval and get speed
                                    # need to get midpoint in next step so we can preserve the interval
                                    #midpoint_list = []
                                    #for counter in np.arange(non_overlap_list.shape[0]):
                                    #    midpoint_list.append((non_overlap_list[counter,0]+non_overlap_list[counter,1])/2)
                                    #arm2_end_timebins = midpoint_list.copy()
                                    arm2_end_timebins = non_overlap_list.copy()

                                    # look up the speed for each midpoint - based on adjusted timestamps
                                    # this should already be in LFPspectogram
                                    arm2_rip_overlap_times = []
                                    arm1_rip_overlap_times = []

                            # somehow need to keep all 1000 shuffles
                            # for shuffle i think do the next matching here
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
                                elif new_speed > speed_thresh and min_diff < 0.03:
                                    random_times_mvt.append(new_time)
                                elif new_speed < speed_thresh and min_diff < 0.03:
                                    random_times_still.append(new_time)


                            # add to list for shuffles
                            all_times_random = len(random_times_mvt)+len(random_times_rip)+len(random_times_still)
                            rip_times_shuf_list.append(len(random_times_rip)/all_times_random)
                            still_times_shuf_list.append(len(random_times_still)/all_times_random)
                            mvt_times_shuf_list.append(len(random_times_mvt)/all_times_random)

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

                    if key["lfp_spect_param_name"] != 'speed_4_random' or key["lfp_spect_param_name"] != 'random_shuffle':
                        for time in process_timebins:

                            if key["lfp_spect_param_name"] == 'whole_post_smooth':
                                midpoint = (time[0]+time[1])/2
                                adjusted_timestamp = midpoint
                            else:
                                adjusted_timestamp = time / 30000 + offset
                            # print(adjusted_timestamp[0])
                            # try to use min_diff here
                            new_time, placeholder, min_diff = find_nearest_1(
                                spyglass_position_df.index, adjusted_timestamp
                            )
                            #print(min_diff,new_time,adjusted_timestamp)
                            new_speed = spyglass_position_df.loc[[new_time]][
                                "speed_smooth"
                            ].values[0]
                            #all_speeds_arm1.append(new_speed)

                            placeholder, placeholder, target_time_diff = find_nearest_1(
                                target_non_rip,new_time
                            )

                            # if not using this, then comment out
                            # NOTE: should only use if this is a close match (min_diff < 0.02)
                            # i think we can just use check above
                            #rip_time, placeholder, min_diff = find_nearest_1(
                            #    lfp_timestamps, adjusted_timestamp)
                            #print(min_diff,adjusted_timestamp,rip_time)

                            if key["lfp_spect_param_name"] == 'whole_post_smooth':
                                if new_speed > speed_thresh and min_diff < 0.02:
                                    arm1_times_mvt.append(time)
                                    #print(len(arm1_times_mvt))
                                elif new_speed < speed_thresh and min_diff < 0.02:
                                    arm1_times_still.append(time)
                                # target times
                                if target_arm == 1 and target_time_diff < 0.03 and new_speed > speed_thresh:
                                    target_mvt.append(time)
                                elif target_arm == 1 and target_time_diff < 0.03 and new_speed < speed_thresh:
                                    target_still.append(time)
                            else:
                                # add time to list for either mvt or still
                                if time in arm1_rip_overlap_times:
                                    arm1_times_rip.append(new_time)
                                elif new_speed > speed_thresh and min_diff < 0.02:
                                    arm1_times_mvt.append(new_time)
                                    #print(len(arm1_times_mvt))
                                elif new_speed < speed_thresh and min_diff < 0.02:
                                    arm1_times_still.append(new_time)

                        if key["lfp_spect_param_name"] == 'whole_post_smooth':
                            print('done with arm 1',len(arm1_end_timebins))
                        else:
                            print(
                                "done with arm 1",
                                arm1_end_timebins.shape,
                            )

                    # add condtional for smoothed decode
                    if smoothing_on == 1:
                        process_timebins = arm2_smooth_midpoint.copy()
                    else:
                        process_timebins = arm2_end_timebins.copy()

                    if key["lfp_spect_param_name"] != 'speed_4_random' or key["lfp_spect_param_name"] != 'random_shuffle':
                        for time in process_timebins:
                            if key["lfp_spect_param_name"] == 'whole_post_smooth':
                                midpoint = (time[0]+time[1])/2
                                adjusted_timestamp = midpoint
                            else:
                                adjusted_timestamp = time / 30000 + offset
                            # print(adjusted_timestamp[0])
                            # try to use min_diff here
                            new_time, placeholder, min_diff = find_nearest_1(
                                spyglass_position_df.index, adjusted_timestamp
                            )
                            new_speed = spyglass_position_df.loc[[new_time]][
                                "speed_smooth"
                            ].values[0]
                            #all_speeds_arm2.append(new_speed)

                            placeholder, placeholder, target_time_diff = find_nearest_1(
                                target_non_rip,new_time
                            )
                            # if not using this, then comment out
                            #rip_time, placeholder, min_diff = find_nearest_1(
                            #    lfp_timestamps, adjusted_timestamp)

                            if key["lfp_spect_param_name"] == 'whole_post_smooth':
                                if new_speed > speed_thresh and min_diff < 0.02:
                                    arm2_times_mvt.append(time)
                                    #print(len(arm1_times_mvt))
                                elif new_speed < speed_thresh and min_diff < 0.02:
                                    arm2_times_still.append(time)
                                # target times
                                if target_arm == 2 and target_time_diff < 0.03 and new_speed > speed_thresh:
                                    target_mvt.append(time)
                                elif target_arm == 2 and target_time_diff < 0.03 and new_speed < speed_thresh:
                                    target_still.append(time)
                            else:
                                # add time to list for either mvt or still
                                if time in arm2_rip_overlap_times:
                                    arm2_times_rip.append(new_time)
                                elif new_speed > speed_thresh and min_diff < 0.02:
                                    arm2_times_mvt.append(new_time)
                                elif new_speed < speed_thresh and min_diff < 0.02:
                                    arm2_times_still.append(new_time)

                        if key["lfp_spect_param_name"] == 'whole_post_smooth':
                            print('done with arm 2',len(arm2_end_timebins))
                        else:
                            print(
                                "arm 2 timebins",
                                arm2_end_timebins.shape,
                            )
                    if key["lfp_spect_param_name"] != 'whole_post_smooth':
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

                    # this needs to run for all 1000 loops - but this will be very slow - try 1000 each
                    if key["lfp_spect_param_name"] != 'random_shuffle' or key["lfp_spect_param_name"] != 'whole_post_smooth':
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
        if key["lfp_spect_param_name"] == 'whole_post_smooth':
            print('arm1. mvt: ',len(arm1_times_mvt),'rip:',len(unique_rip_arm1),'still:',len(arm1_times_still))
            print('arm2. mvt: ',len(arm2_times_mvt),'rip:',len(unique_rip_arm2),'still:',len(arm2_times_still))
            #print('random. mvt: ',len(random_times_mvt),'rip:',len(random_times_rip),'still:',len(random_times_still))
        else:
            print('arm1. mvt: ',len(arm1_times_mvt),'rip:',len(arm1_times_rip),'still:',len(arm1_times_still))
            print('arm2. mvt: ',len(arm2_times_mvt),'rip:',len(arm2_times_rip),'still:',len(arm2_times_still))
            print('random. mvt: ',len(random_times_mvt),'rip:',len(random_times_rip),'still:',len(random_times_still))


        if smoothing_on == 1:
            key['all_arm1_times'] = arm1_smooth_midpoint.shape[0]
            key['all_arm2_times'] = arm2_smooth_midpoint.shape[0]
            key['all_random_times'] = center_well_timebins_sub.shape[0]
        elif key["lfp_spect_param_name"] == 'whole_post_smooth':
            key['all_arm1_times'] = len(arm1_end_timebins)
            key['all_arm2_times'] = len(arm2_end_timebins)
            key['all_random_times'] = 0
        else:
            key['all_arm1_times'] = arm1_end_timebins.shape[0]
            key['all_arm2_times'] = arm2_end_timebins.shape[0]
            key['all_random_times'] = center_well_timebins_sub.shape[0]
        key["arm1_times_mvt"] = arm1_times_mvt
        if key["lfp_spect_param_name"] == 'whole_post_smooth':
            key["arm1_times_rip"] = unique_rip_arm1
        else:
            key["arm1_times_rip"] = arm1_times_rip
        key["arm1_times_still"] = arm1_times_still
        key["arm2_times_mvt"] = arm2_times_mvt
        if key["lfp_spect_param_name"] == 'whole_post_smooth':
            key["arm2_times_rip"] = unique_rip_arm2
        else:
            key["arm2_times_rip"] = arm2_times_rip
        key["arm2_times_still"] = arm2_times_still

        # change what is saved based on shuffle or not
        if key["lfp_spect_param_name"] == 'random_shuffle':
            key["random_times_mvt"] = mvt_times_shuf_list
            key["random_times_rip"] = rip_times_shuf_list
            key["random_times_still"] = still_times_shuf_list
        else:
            key["random_times_mvt"] = random_times_mvt
            key["random_times_rip"] = random_times_rip
            key["random_times_still"] = random_times_still
        key["reward_times_mvt"] = target_mvt
        key["reward_times_rip"] = target_ripple
        key["reward_times_still"] = target_still

        self.insert1(key)
