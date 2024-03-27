# required packages:
import datajoint as dj
import pandas as pd
import sys

from spyglass.common import (
    IntervalList,
    Nwbfile,
    IntervalLinearizedPosition,
    IntervalPositionInfo,
)
from spyglass.spikesorting.v0 import (
    Waveforms,
    CuratedSpikeSorting,
    SpikeSorting,
    SortInterval,
)
from spyglass.utils.dj_helper_fn import fetch_nwb
from hippocampus_content_neurofeedback.mcoulter_realtime_filename import (
    RealtimeFilename,
)
from scipy import ndimage
from spyglass.mcoulter_statescript_rewards import StatescriptReward
from spyglass.mcoulter_statescript_time import StatescriptTaskTime

import pprint

import warnings
import numpy as np
import xarray as xr
import logging
import datetime
import statistics
import os
import pickle
from pynwb import NWBFile, TimeSeries, NWBHDF5IO


FORMAT = "%(asctime)s %(message)s"

logging.basicConfig(level="INFO", format=FORMAT, datefmt="%d-%b-%y %H:%M:%S")
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

schema = dj.schema("mcoulter_reward_cluster_list")


# parameters
@schema
class RewardClusterListParameters(dj.Manual):
    definition = """
    reward_cluster_list_param_name : varchar(500)
    ---
    reward_cluster_list_parameters : blob
    """

    # can include description above
    # description : varchar(500)

    def insert_default(self):
        reward_cluster_list_parameters = {}
        reward_cluster_list_parameters["interneuron_FR"] = 7
        reward_cluster_list_parameters["min_spikes"] = 100
        reward_cluster_list_parameters["baseline_time"] = 20
        reward_cluster_list_parameters["reward_time"] = 20
        reward_cluster_list_parameters["FR_increase"] = 4
        reward_cluster_list_parameters["min_trial_frac"] = 0.1
        self.insert1(["default", reward_cluster_list_parameters], skip_duplicates=True)


# select parameters and cluster
@schema
class RewardClusterListSelection(dj.Manual):
    definition = """
    -> Nwbfile
    -> RewardClusterListParameters
    ---
    """


# this is the computed table - basically it has the results of the analysis
@schema
class RewardClusterList(dj.Computed):
    definition = """
    -> RewardClusterListSelection
    ---
    reward_cluster_ptsh : mediumblob
    reward_cluster_dict : mediumblob

    """

    def make(self, key):
        print(f"finding reward clusters for: {key}")

        # these are all in the paramters set now
        reward_cluster_parameters = (
            RewardClusterListParameters
            & {"reward_cluster_list_param_name": key["reward_cluster_list_param_name"]}
        ).fetch()[0][1]
        interneuron_FR = reward_cluster_parameters["interneuron_FR"]
        min_spikes = reward_cluster_parameters["min_spikes"]
        baseline_time = reward_cluster_parameters["baseline_time"]
        reward_time = reward_cluster_parameters["reward_time"]
        FR_increase = reward_cluster_parameters["FR_increase"]
        min_trial_frac = reward_cluster_parameters["min_trial_frac"]
        no_enrich_clusters = reward_cluster_parameters['no_enrich_clusters']

        if no_enrich_clusters == 1:
            print('cluster list without FR increase')

        # this seems to work now - should work for any session

        # create firing area dict: to find clusters active +/- 1 sec around reward

        all_session_filename_dict = {}
        cluster_summary_dict = {}

        all_session_cluster_summary_table = np.zeros((100, 5))
        session_counter = -1
        curation_id = 1

        reward_ptsh_histogram_dict = {}
        new_reward_cell_dict = {}
        reward_cell_dict_v2 = {}

        FR_change_list_content = []
        FR_change_list_head = []

        interval_list = (IntervalList & {"nwb_file_name": key["nwb_file_name"]}).fetch(
            "interval_list_name"
        )

        reward_ptsh_histogram_dict[key["nwb_file_name"]] = {}
        new_reward_cell_dict[key["nwb_file_name"]] = {}
        reward_cell_dict_v2[key["nwb_file_name"]] = {}

        # NOTE: for ginny tet 34 and 35 dont work because they have a dead channel included
        # for session in ['run2_v_run3']:
        for session in ["run1_v_run2", "run2_v_run3"]:
            good_cluster_list = []

            print(key["nwb_file_name"], session)

            # tet_list = all_tet_list.copy()
            tet_list = np.unique(
                (
                    SpikeSorting
                    & {"nwb_file_name": key["nwb_file_name"]}
                    & {'artifact_removed_interval_list_name LIKE "%100_prop%"'}
                    & {"sorter": "mountainsort4"}
                ).fetch("sort_group_id")
            )

            # get correct sort interval and position intervals
            sort_intervals = np.unique(
                (
                    Waveforms
                    & {"nwb_file_name": key["nwb_file_name"]}
                    & {"waveform_params_name": "default_whitened"}
                    & {'artifact_removed_interval_list_name LIKE "%100_prop%"'}
                ).fetch("sort_interval_name")
            )

            # now need to parse this to get the position intervals
            if session == "run1_v_run2":
                sort_interval = sort_intervals[0]
                # 1st session
                for interval in interval_list:
                    if interval[0] == "0":
                        if sort_intervals[0].split("_")[0] in interval:
                            int_number = np.int(interval.split("_")[0])
                            print(int_number)
                            first_sess = "pos " + str(int_number - 1) + " valid times"
                # 2nd session
                for interval in interval_list:
                    if interval[0] == "0":
                        if sort_intervals[0].split("_")[1] in interval:
                            int_number = np.int(interval.split("_")[0])
                            print(int_number)
                            second_sess = "pos " + str(int_number - 1) + " valid times"

            elif session == "run2_v_run3":
                # tet_list = np.unique((PlaceField2 & {'nwb_file_name' : nwb_file_name} &
                # {'artifact_removed_interval_list_name LIKE "%100_prop_01_2ms%"'} &
                # {'interval_list_name': 'pos 5 valid times'}
                #   & {'place_field_param_name': '5cm_speed'}).fetch('sort_group_id'))
                sort_interval = sort_intervals[1]
                # 1st session
                for interval in interval_list:
                    if interval[0] == "0":
                        if sort_intervals[1].split("_")[0] in interval:
                            int_number = np.int(interval.split("_")[0])
                            print(int_number)
                            first_sess = "pos " + str(int_number - 1) + " valid times"
                # 2nd session
                for interval in interval_list:
                    if interval[0] == "0":
                        if sort_intervals[1].split("_")[1] in interval:
                            int_number = np.int(interval.split("_")[0])
                            print(int_number)
                            second_sess = "pos " + str(int_number - 1) + " valid times"

            # make one dict for each run session
            reward_ptsh_histogram_dict[key["nwb_file_name"]][sort_interval] = {}
            reward_ptsh_histogram_dict[key["nwb_file_name"]][sort_interval][
                first_sess
            ] = {}
            reward_ptsh_histogram_dict[key["nwb_file_name"]][sort_interval][
                second_sess
            ] = {}
            # reward_ptsh_histogram_dict[key["nwb_file_name"]][key[sort_interval_name]]["run3"] = {}

            new_reward_cell_dict[key["nwb_file_name"]][sort_interval] = {}
            new_reward_cell_dict[key["nwb_file_name"]][sort_interval][first_sess] = {}
            new_reward_cell_dict[key["nwb_file_name"]][sort_interval][second_sess] = {}
            # new_reward_cell_dict[key["nwb_file_name"]][key[sort_interval_name]]["run3"] = {}

            reward_cell_dict_v2[key["nwb_file_name"]][sort_interval] = {}
            reward_cell_dict_v2[key["nwb_file_name"]][sort_interval][first_sess] = {}
            reward_cell_dict_v2[key["nwb_file_name"]][sort_interval][second_sess] = {}

            session_counter += 1

            # what is this?
            all_session_filename_dict[
                key["nwb_file_name"] + "_" + session
            ] = session_counter

            print("tet list", tet_list)
            # print('tet list new',tet_list_new)

            # NOTE: need to subtract 0.045 sec for the decoder delay to statescript

            reward_times = (
                StatescriptReward()
                & {"nwb_file_name": key["nwb_file_name"]}
                & {"interval_list_name": first_sess}
            ).fetch("reward_times") - 0.045
            print("1st sess reward count:", reward_times[0].shape[0], first_sess)

            if (
                key["nwb_file_name"] == "ron20210823_.nwb"
                and second_sess == "pos 6 valid times"
            ):
                reward_times_2nd = (
                    StatescriptReward()
                    & {"nwb_file_name": key["nwb_file_name"]}
                    & {"interval_list_name": "pos 5 valid times"}
                ).fetch("reward_times") - 0.045
                print(
                    "2nd sess reward count:", reward_times_2nd[0].shape[0], second_sess
                )
            else:
                reward_times_2nd = (
                    StatescriptReward()
                    & {"nwb_file_name": key["nwb_file_name"]}
                    & {"interval_list_name": second_sess}
                ).fetch("reward_times") - 0.045
                print(
                    "2nd sess reward count:", reward_times_2nd[0].shape[0], second_sess
                )

            spike_time_data = dict()

            for single_session in [first_sess, second_sess]:
                #### NEW #####
                # position info
                linear_pos = (
                    IntervalLinearizedPosition()
                    & {"nwb_file_name": key["nwb_file_name"]}
                    & {"interval_list_name": single_session}
                    & {"position_info_param_name": "default_decoding"}
                ).fetch1_dataframe()

                # update this to just be time at center well - below
                linear_pos_box = linear_pos[linear_pos["linear_position"] < 60]

                # find times at center well
                rat_name = key["nwb_file_name"].split("2")[0]

                # center: [449, 330], or [634,648]
                if rat_name == "pippin":
                    print("pippin session")
                    center_well = [634, 648]
                else:
                    center_well = [449, 330]

                pos_table = (
                    IntervalPositionInfo()
                    & {"nwb_file_name": key["nwb_file_name"]}
                    & {"interval_list_name": single_session}
                    & {"position_info_param_name": "default_decoding"}
                ).fetch1_dataframe()
                pos_table_dist = pos_table.copy()
                pos_table_dist["center_x"] = center_well[0] * 0.22
                pos_table_dist["center_y"] = center_well[1] * 0.22
                pos_table_dist["distance"] = np.sqrt(
                    np.square(
                        pos_table_dist["center_x"] - pos_table_dist["head_position_x"]
                    )
                    + np.square(
                        pos_table_dist["center_y"] - pos_table_dist["head_position_y"]
                    )
                )
                center_pos = pos_table_dist[pos_table_dist["distance"] < 17]
                print("found center well spikes")

                # first define as begin and end of session in case task2 landmarks are missing

                task2_start = (
                    StatescriptTaskTime
                    & {"nwb_file_name": key["nwb_file_name"]}
                    & {"interval_list_name": single_session}
                ).fetch("task2_start")
                task2_end = (
                    StatescriptTaskTime
                    & {"nwb_file_name": key["nwb_file_name"]}
                    & {"interval_list_name": single_session}
                ).fetch("task3_start")
                if task2_end == 0:
                    print("no task 3")
                    task2_end = (
                        StatescriptTaskTime
                        & {"nwb_file_name": key["nwb_file_name"]}
                        & {"interval_list_name": single_session}
                    ).fetch("end_time")

                print("task 2 time:", (task2_end - task2_start) / 60)

                print(session, single_session)
                for tetrode in tet_list:
                    # for ginny
                    # for tetrode in tet_list_new:
                    # print('tetrode:',sort_group)
                    spike_time_data[tetrode] = {}
                    # reward_ptsh_histogram_dict[key['nwb_file_name']][session]['run1'][tetrode] = {}
                    # reward_ptsh_histogram_dict[key['nwb_file_name']][session]['run2'][tetrode] = {}
                    # reward_ptsh_histogram_dict[key['nwb_file_name']][session]['run3'][tetrode] = {}

                    # try:
                    obj_string = (
                        CuratedSpikeSorting
                        & {"nwb_file_name": key["nwb_file_name"]}
                        & {"sort_interval_name": sort_interval}
                        & {'artifact_removed_interval_list_name LIKE "%100_prop%"'}
                        & {"sorter": "mountainsort4"}
                        & {"sort_group_id": tetrode}
                    ).fetch("units_object_id")[0]
                    # print(tetrode, obj_string)
                    if len(obj_string) > 0:
                        # print("good tetrode", tetrode)
                        curation_versions = (
                            CuratedSpikeSorting
                            & {"nwb_file_name": key["nwb_file_name"]}
                            & {"sort_interval_name": sort_interval}
                            & {'artifact_removed_interval_list_name LIKE "%100_prop%"'}
                            & {"sorter": "mountainsort4"}
                            & {"sort_group_id": tetrode}
                        ).fetch("curation_id")

                        # unit_obj_name = (
                        #    CuratedSpikeSorting
                        #    & {"nwb_file_name": key["nwb_file_name"]}
                        #    & {"sort_interval_name": sort_interval}
                        #    & {'artifact_removed_interval_list_name LIKE "%100_prop%"'}
                        #    & {"sorter": "mountainsort4"}
                        #    & {"sort_group_id": tetrode}
                        # ).fetch_nwb("units_object_id")[0]

                        if 3 in curation_versions:
                            print("manually curated tetrode", tetrode)
                            obj_string = (
                                CuratedSpikeSorting
                                & {"nwb_file_name": key["nwb_file_name"]}
                                & {"sort_interval_name": sort_interval}
                                & {
                                    'artifact_removed_interval_list_name LIKE "%100_prop%"'
                                }
                                & {"sorter": "mountainsort4"}
                                & {"sort_group_id": tetrode}
                                & {"curation_id": 3}
                            ).fetch("units_object_id")[0]
                            # print(tetrode, obj_string)
                            if len(obj_string) > 0:
                                print("good manual curated tetrode")
                                tetrode_spike_times = (
                                    CuratedSpikeSorting
                                    & {"nwb_file_name": key["nwb_file_name"]}
                                    & {"sort_interval_name": sort_interval}
                                    & {
                                        'artifact_removed_interval_list_name LIKE "%100_prop%"'
                                    }
                                    & {"sorter": "mountainsort4"}
                                    & {"curation_id": 3}
                                    & {"sort_group_id": tetrode}
                                ).fetch_nwb()[0]["units"]["spike_times"]
                                valid_unit_ids = (
                                    CuratedSpikeSorting.Unit
                                    & {"nwb_file_name": key["nwb_file_name"]}
                                    & {"sort_interval_name": sort_interval}
                                    & {
                                        'artifact_removed_interval_list_name LIKE "%100_prop%"'
                                    }
                                    & {"sorter": "mountainsort4"}
                                    & {"curation_id": 3}
                                    & {"sort_group_id": tetrode}
                                ).fetch("unit_id")
                                curation_id = 3

                        elif 1 in curation_versions:
                            # print("regular tetrode", tetrode)
                            tetrode_spike_times = (
                                CuratedSpikeSorting
                                & {"nwb_file_name": key["nwb_file_name"]}
                                & {"sort_interval_name": sort_interval}
                                & {
                                    'artifact_removed_interval_list_name LIKE "%100_prop%"'
                                }
                                & {"sorter": "mountainsort4"}
                                & {"curation_id": 1}
                                & {"sort_group_id": tetrode}
                            ).fetch_nwb()[0]["units"]["spike_times"]
                            valid_unit_ids = (
                                CuratedSpikeSorting.Unit
                                & {"nwb_file_name": key["nwb_file_name"]}
                                & {"sort_interval_name": sort_interval}
                                & {
                                    'artifact_removed_interval_list_name LIKE "%100_prop%"'
                                }
                                & {"sorter": "mountainsort4"}
                                & {"curation_id": 1}
                                & {"sort_group_id": tetrode}
                            ).fetch("unit_id")
                            curation_id = 1

                        for cluster in valid_unit_ids:
                            # print(tetrode, cluster)
                            spike_time_data[tetrode][cluster] = tetrode_spike_times[
                                cluster
                            ]
                            # print(tetrode, cluster)

                            # loop through each cluster and find ones with enough spiking

                            # during taskState1: mean firing rate < 7
                            # during taskState1: >50 spikes
                            # or these could be filters for whole time ?

                            start_time = (
                                IntervalList
                                & {"nwb_file_name": key["nwb_file_name"]}
                                & {"interval_list_name": single_session}
                            ).fetch("valid_times")[0][0][0]
                            end_time = (
                                IntervalList
                                & {"nwb_file_name": key["nwb_file_name"]}
                                & {"interval_list_name": single_session}
                            ).fetch("valid_times")[0][0][1]
                            session_duration = end_time - start_time
                            session_spikes = tetrode_spike_times[cluster][
                                (tetrode_spike_times[cluster] > start_time)
                                & (tetrode_spike_times[cluster] < end_time)
                            ]
                            all_spikes = tetrode_spike_times[cluster]

                            whole_firing_rate = (
                                session_spikes.shape[0] / session_duration
                            )

                            if single_session == first_sess:
                                all_reward_spikes = np.zeros((reward_times[0].shape[0],))
                            elif single_session == second_sess:
                                all_reward_spikes = np.zeros((reward_times_2nd[0].shape[0],))
                            
                            # NOTE: for content vs head direction, we need to compare all clusters at reward times

                            # need this try because of the issue where some curation id 3 tets have no clusters
                            try:
                                # remove MUA units
                                cluster_label = (
                                    CuratedSpikeSorting.Unit
                                    & {"nwb_file_name": key["nwb_file_name"]}
                                    & {"sort_interval_name": sort_interval}
                                    & {
                                        'artifact_removed_interval_list_name LIKE "%100_prop%"'
                                    }
                                    & {"sorter": "mountainsort4"}
                                    & {"curation_id": curation_id}
                                    & {"sort_group_id": tetrode}
                                    & {"unit_id": cluster}
                                ).fetch("label")[0]
                                if cluster_label == "mua":
                                    pass
                                    # print("mua")
                                elif cluster_label == "":
                                    # check for interneuron and min spike count
                                    # good clusters: find histogram near reward times
                                    if (
                                        whole_firing_rate < interneuron_FR
                                        and session_spikes.shape[0] > min_spikes
                                    ):
                                        # print("good cluster")
                                        # want to try to include more clusters
                                        # save total spikes 100 msec before reward
                                        # save spikes per trial

                                        # make a histogram with 10msec bins then aerage histogram over 75 rewards
                                        reward_time_counter = 0
                                        
                                        if single_session == first_sess:
                                            session_reward_times = reward_times[
                                                0
                                            ].copy()
                                        elif single_session == second_sess:
                                            session_reward_times = reward_times_2nd[
                                                0
                                            ].copy()
                                        cluster_reward_spikes = np.zeros((session_reward_times.shape[0],))

                                        reward_spike_count = 0
                                        for time_of_reward in session_reward_times:
                                            nearby_spikes = (
                                                session_spikes[
                                                    (
                                                        session_spikes
                                                        > time_of_reward - 1
                                                    )
                                                    & (
                                                        session_spikes
                                                        < time_of_reward + 1
                                                    )
                                                ]
                                                - time_of_reward
                                            )
                                            reward_spike_count += all_spikes[
                                                (
                                                    all_spikes
                                                    > (
                                                        time_of_reward
                                                        - reward_time / 100
                                                    )
                                                )
                                                & (all_spikes < time_of_reward)
                                            ].shape[0]
                                            # use reward_time_counter?
                                            cluster_reward_spikes[reward_time_counter] = session_spikes[(session_spikes>(time_of_reward - reward_time/100))
                                                             & (session_spikes<time_of_reward)].shape[0]

                                            firing_hist = np.histogram(
                                                nearby_spikes,
                                                bins=np.arange(-1, 1, 0.01),
                                            )[0]

                                            if reward_time_counter == 0:
                                                # all_reward_firing_array = np.array(firing_hist)
                                                all_reward_firing_array = np.array(
                                                    firing_hist
                                                )
                                            else:
                                                all_reward_firing_array = np.vstack(
                                                    (
                                                        all_reward_firing_array,
                                                        firing_hist,
                                                    )
                                                )
                                            reward_time_counter += 1
                                        all_reward_spikes += cluster_reward_spikes

                                        if single_session == first_sess:
                                            # count next session reward spikes
                                            reward_spike_count_2nd = 0
                                            for time_of_reward_2 in reward_times_2nd[0]:
                                                reward_spike_count_2nd += all_spikes[
                                                    (
                                                        all_spikes
                                                        > (
                                                            time_of_reward_2
                                                            - reward_time / 100
                                                        )
                                                    )
                                                    & (all_spikes < time_of_reward_2)
                                                ].shape[0]

                                        # print(all_reward_firing_array)
                                        firing_average = np.mean(
                                            all_reward_firing_array, axis=0
                                        )
                                        # firing_st_dev = np.std(all_reward_firing_array,axis=0)

                                        # save histogram to dict
                                        if (
                                            single_session == first_sess
                                            and session == "run1_v_run2"
                                        ):
                                            reward_ptsh_histogram_dict[
                                                key["nwb_file_name"]
                                            ][sort_interval][first_sess][
                                                str(curation_id)
                                                + "_"
                                                + str(tetrode)
                                                + "_"
                                                + str(cluster)
                                            ] = firing_average
                                        elif (
                                            single_session == first_sess
                                            and session == "run2_v_run3"
                                        ):
                                            reward_ptsh_histogram_dict[
                                                key["nwb_file_name"]
                                            ][sort_interval][first_sess][
                                                str(curation_id)
                                                + "_"
                                                + str(tetrode)
                                                + "_"
                                                + str(cluster)
                                            ] = firing_average
                                        elif (
                                            single_session == second_sess
                                            and session == "run2_v_run3"
                                        ):
                                            reward_ptsh_histogram_dict[
                                                key["nwb_file_name"]
                                            ][sort_interval][second_sess][
                                                str(curation_id)
                                                + "_"
                                                + str(tetrode)
                                                + "_"
                                                + str(cluster)
                                            ] = firing_average
                                        elif (
                                            single_session == second_sess
                                            and session == "run1_v_run2"
                                        ):
                                            reward_ptsh_histogram_dict[
                                                key["nwb_file_name"]
                                            ][sort_interval][second_sess][
                                                str(curation_id)
                                                + "_"
                                                + str(tetrode)
                                                + "_"
                                                + str(cluster)
                                            ] = firing_average

                                        # now we should do the next calculation (from cell below)
                                        sigma = 1
                                        time_smooth = ndimage.gaussian_filter1d(
                                            np.arange(199), sigma * 1
                                        )
                                        actual_smooth = ndimage.gaussian_filter1d(
                                            firing_average, sigma * 1
                                        )

                                        # spikes 200 msec before reward
                                        FR_before_reward = np.average(
                                            firing_average[100 - reward_time : 100]
                                            / 0.01
                                        )
                                        # spikes 1000-800 msec before reward
                                        FR_baseline = np.average(
                                            firing_average[0:baseline_time] / 0.01
                                        )

                                        # content vs head direction: use absolute FR change
                                        FR_abs_change = FR_before_reward - FR_baseline
                                        FR_change_list_content.append(FR_abs_change)

                                        # reward cells: use FR increase
                                        FR_increase_cluster = (
                                            FR_before_reward / FR_baseline
                                        )

                                        # this is for use with no_enrich
                                        # save before FR check
                                        if (single_session == first_sess and session == "run1_v_run2"):
                                            reward_cell_dict_v2[
                                                key["nwb_file_name"]][sort_interval][first_sess][str(
                                                    curation_id)+ "_"+ str(tetrode)+ "_"+ str(cluster)
                                                ] = all_reward_spikes
                                        elif (single_session == first_sess and session == "run2_v_run3"):
                                            reward_cell_dict_v2[
                                                key["nwb_file_name"]][sort_interval][first_sess][str(
                                                    curation_id)+ "_"+ str(tetrode)+ "_"+ str(cluster)
                                                ] = all_reward_spikes
                                        elif (single_session == second_sess and session == "run1_v_run2"):
                                            reward_cell_dict_v2[
                                                key["nwb_file_name"]][sort_interval][second_sess][str(
                                                    curation_id)+ "_"+ str(tetrode)+ "_"+ str(cluster)
                                                ] = all_reward_spikes
                                        elif (single_session == second_sess and session == "run2_v_run3"):
                                            reward_cell_dict_v2[
                                                key["nwb_file_name"]][sort_interval][second_sess][str(
                                                    curation_id)+ "_"+ str(tetrode)+ "_"+ str(cluster)
                                                ] = all_reward_spikes
                                        #print(tetrode,all_reward_spikes)
                                        
                                        # original: FR_increase >4 and max(actual_smooth)>0.05
                                        if (
                                            FR_increase_cluster > FR_increase
                                            and max(actual_smooth) > min_trial_frac
                                        ):
                                            # plt.figure(figsize=(4,3))
                                            # plt.plot(time_smooth,actual_smooth)
                                            # plt.text(150,0.28,'FR center')
                                            # plt.text(150,0.25,FR_center_1)
                                            # plt.text(150,0.22,FR_center_2)
                                            # plt.title('run 1. tetrode '+str(item[0])+' cluster '+str(item2[0]))
                                            # plt.ylim(-0.05,0.3)
                                            # print('reward cluster',tetrode,cluster,
                                            #              np.around(FR_increase,decimals=3),
                                            #              np.around(max(actual_smooth),decimals=3))

                                            # save reward cell dictionary
                                            print(
                                                "reward cluster",
                                                curation_id,
                                                tetrode,
                                                cluster,
                                            )

                                            # find task2 center spikes

                                            #### NEW #####
                                            # print(tetrode_spike_times[np.int(cluster)].shape)
                                            all_spike_count = session_spikes.shape[0]
                                            # all_spikes = session_spikes
                                            # only task 2
                                            try:
                                                task2_spikes = session_spikes[
                                                    (session_spikes > task2_start)
                                                    & (session_spikes < task2_end)
                                                ]
                                                # only task2 in box area (tolerance = 50 msec)
                                                task2_box_spikes = np.array(
                                                    list(
                                                        {
                                                            i
                                                            for i in task2_spikes
                                                            if np.isclose(
                                                                linear_pos_box.index,
                                                                i,
                                                                0,
                                                                0.05,
                                                            ).any()
                                                        }
                                                    )
                                                )

                                                task2_center_spikes = np.array(
                                                    list(
                                                        {
                                                            i
                                                            for i in task2_spikes
                                                            if np.isclose(
                                                                center_pos.index, i, 0, 0.05
                                                            ).any()
                                                        }
                                                    )
                                                )
                                            except ValueError as e:
                                                print('no task2')
                                                task2_center_spikes = [0]
                                        
                                            # single_session_dict[item_count] = tetrode_spike_times[np.int(cluster)]
                                            # single_session_dict[item_count] = task2_box_spikes

                                            # print(
                                            #    "all spikes",
                                            #    all_spike_count,
                                            #    "task2 spikes",
                                            #    task2_spikes.shape,
                                            #    "box spikes",
                                            #    task2_box_spikes.shape,
                                            #    "center_spikes",
                                            #    task2_center_spikes.shape,
                                            # )
                                            # item_count += 1

                                            if (
                                                single_session == first_sess
                                                and session == "run1_v_run2"
                                            ):
                                                new_reward_cell_dict[
                                                    key["nwb_file_name"]
                                                ][sort_interval][first_sess][
                                                    str(curation_id)
                                                    + "_"
                                                    + str(tetrode)
                                                    + "_"
                                                    + str(cluster)
                                                ] = [
                                                    FR_increase_cluster,
                                                    max(actual_smooth),
                                                    reward_spike_count,
                                                    reward_times[0].shape[0],
                                                    reward_spike_count_2nd,
                                                    reward_times_2nd[0].shape[0],
                                                    task2_center_spikes,
                                                ]
                                                # for no FR increase
                                                #reward_cell_dict_v2[key["nwb_file_name"]][sort_interval]
                                                #[first_sess][str(curation_id)+ "_"+ str(tetrode)+ "_"+ str(cluster)
                                                #] = all_reward_spikes
                                                
                                            elif (
                                                single_session == first_sess
                                                and session == "run2_v_run3"
                                            ):
                                                new_reward_cell_dict[
                                                    key["nwb_file_name"]
                                                ][sort_interval][first_sess][
                                                    str(curation_id)
                                                    + "_"
                                                    + str(tetrode)
                                                    + "_"
                                                    + str(cluster)
                                                ] = [
                                                    FR_increase_cluster,
                                                    max(actual_smooth),
                                                    reward_spike_count,
                                                    reward_times[0].shape[0],
                                                    reward_spike_count_2nd,
                                                    reward_times_2nd[0].shape[0],
                                                    task2_center_spikes,
                                                ]
                                                # for no FR increase
                                                #reward_cell_dict_v2[key["nwb_file_name"]][sort_interval]
                                                #[first_sess][str(curation_id)+ "_"+ str(tetrode)+ "_"+ str(cluster)
                                                #] = all_reward_spikes
                                            elif (
                                                single_session == second_sess
                                                and session == "run2_v_run3"
                                            ):
                                                new_reward_cell_dict[
                                                    key["nwb_file_name"]
                                                ][sort_interval][second_sess][
                                                    str(curation_id)
                                                    + "_"
                                                    + str(tetrode)
                                                    + "_"
                                                    + str(cluster)
                                                ] = [
                                                    FR_increase_cluster,
                                                    max(actual_smooth),
                                                    reward_spike_count,
                                                    reward_times[0].shape[0],
                                                    reward_spike_count_2nd,
                                                    reward_times_2nd[0].shape[0],
                                                    task2_center_spikes,
                                                ]
                                                # for no FR increase
                                                #reward_cell_dict_v2[key["nwb_file_name"]][sort_interval]
                                                #[second_sess][str(curation_id)+ "_"+ str(tetrode)+ "_"+ str(cluster)
                                                #] = all_reward_spikes
                                            elif (
                                                single_session == second_sess
                                                and session == "run1_v_run2"
                                            ):
                                                new_reward_cell_dict[
                                                    key["nwb_file_name"]
                                                ][sort_interval][second_sess][
                                                    str(curation_id)
                                                    + "_"
                                                    + str(tetrode)
                                                    + "_"
                                                    + str(cluster)
                                                ] = [
                                                    FR_increase_cluster,
                                                    max(actual_smooth),
                                                    reward_spike_count,
                                                    reward_times[0].shape[0],
                                                    reward_spike_count_2nd,
                                                    reward_times_2nd[0].shape[0],
                                                    task2_center_spikes,
                                                ]
                                                # for no FR increase
                                                #reward_cell_dict_v2[key["nwb_file_name"]][sort_interval]
                                                #[second_sess][str(curation_id)+ "_"+ str(tetrode)+ "_"+ str(cluster)
                                                #] = all_reward_spikes
                                            # try to plot the raster - looks good
                                            # note this counts all spikes in the plot window
                                            # trial_count = session_reward_times.shape[0]
                                            # spike_count = 0
                                            # plt.figure()
                                            # for i in np.arange(session_reward_times.shape[0]):
                                            #    nearby_spikes = (session_spikes[(session_spikes>session_reward_times[i]-1)&
                                            #                    (session_spikes<session_reward_times[i]+1)]-session_reward_times[i])
                                            #    spike_count += nearby_spikes.shape[0]
                                            #    plt.scatter(nearby_spikes,np.repeat(trial_count-i,len(nearby_spikes)),s=5,c='k')
                                            # plt.vlines(0,0,trial_count)
                                            # plt.xlim(-1,1)
                                            # plt.title(nwb_file_name+' tetrode: '+str(tetrode)+' cluster: '+str(cluster)
                                            #          +' spikes: '+str(spike_count))
                                            # save figure
                                            # raster_title = nwb_file_name+'_'+str(tetrode)+'_'+str(cluster)
                                            # plt.savefig('/home/mcoulter/content_feedback_figs/'+raster_title+'_raster.svg',format="svg")
                                else:
                                    print(cluster_label)
                            except IndexError as e:
                                print("curation id 3, no units on tetrode", tetrode)
                        #if single_session == first_sess:
                        #    first_sess_reward_spikes = all_reward_spikes.copy()
                        #    print(first_sess_reward_spikes)
                        #elif single_session == second_sess:
                        #    second_sess_reward_spikes = all_reward_spikes.copy()
                        
                    else:
                        print("no units on tetrode", tetrode)

                #if (single_session == first_sess and session == "run1_v_run2"):
                #    reward_cell_dict_v2[
                #        key["nwb_file_name"]][sort_interval][first_sess] = first_sess_reward_spikes
                #elif (single_session == first_sess and session == "run2_v_run3"):
                #    reward_cell_dict_v2[
                #        key["nwb_file_name"]][sort_interval][first_sess] = first_sess_reward_spikes
                #elif (single_session == second_sess and session == "run1_v_run2"):
                #    reward_cell_dict_v2[
                #        key["nwb_file_name"]][sort_interval][second_sess] = second_sess_reward_spikes
                #elif (single_session == second_sess and session == "run2_v_run3"):
                #    reward_cell_dict_v2[
                #        key["nwb_file_name"]][sort_interval][second_sess] = second_sess_reward_spikes
                
                #reward_cell_dict_v2 = {}
                #reward_cell_dict_v2['entry'] = [first_sess_reward_spikes,
                #                                second_sess_reward_spikes]
        # NOTE: item to save is the dictionary
        # NOTE: need to add curation id to each dictionary entry
        key["reward_cluster_ptsh"] = reward_ptsh_histogram_dict
        #key["reward_cluster_dict"] = new_reward_cell_dict
        # reward clusters with no restrictions
        if no_enrich_clusters == 1:
            key["reward_cluster_dict"] = reward_cell_dict_v2
        else:
            key["reward_cluster_dict"] = new_reward_cell_dict
        self.insert1(key)
