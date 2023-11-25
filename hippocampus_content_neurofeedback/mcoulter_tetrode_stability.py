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
from spyglass.spikesorting import (
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

schema = dj.schema("mcoulter_tetrode_stability")


# parameters
@schema
class TetrodeStabilityParameters(dj.Manual):
    definition = """
    tetrode_stability_param_name : varchar(500)
    ---
    tetrode_stability_parameters : blob
    """

    # can include description above
    # description : varchar(500)

    def insert_default(self):
        tetrode_stability_parameters = {}
        tetrode_stability_parameters["interneuron_FR"] = 7
        tetrode_stability_parameters["min_spikes"] = 100
        self.insert1(
            ["default", tetrode_stability_parameters], skip_duplicates=True
        )


# select parameters and cluster
@schema
class TetrodeStabilitySelection(dj.Manual):
    definition = """
    -> SortInterval
    -> TetrodeStabilityParameters
    ---
    """


# this is the computed table - basically it has the results of the analysis
@schema
class TetrodeStability(dj.Computed):
    definition = """
    -> TetrodeStabilitySelection
    ---
    first_sess_spikes : double
    second_sess_spikes : double
    fraction_first_spikes : blob

    """

    def make(self, key):
        print(f"calculating tetrode stability for: {key}")

        stability_parameters = (
            TetrodeStabilityParameters
            & {
                "tetrode_stability_param_name": key[
                    "tetrode_stability_param_name"
                ]
            }
        ).fetch()[0][1]
        interneuron_FR = stability_parameters["interneuron_FR"]
        min_spikes = stability_parameters["min_spikes"]

        all_session_filename_dict = {}
        cluster_summary_dict = {}

        all_session_cluster_summary_table = np.zeros((100, 5))
        session_counter = -1
        curation_id = 1

        reward_ptsh_histogram_dict = {}
        new_reward_cell_dict = {}

        FR_change_list_content = []
        FR_change_list_head = []

        spike_imbalance_list = []

        interval_list = (
            IntervalList & {"nwb_file_name": key["nwb_file_name"]}
        ).fetch("interval_list_name")

        reward_ptsh_histogram_dict[key["nwb_file_name"]] = {}
        new_reward_cell_dict[key["nwb_file_name"]] = {}

        # NOTE: for ginny tet 34 and 35 dont work because they have a dead channel included
        # for session in ['run2_v_run3']:

        good_cluster_list = []

        # tet_list = all_tet_list.copy()
        tet_list = np.unique(
            (
                SpikeSorting
                & {"nwb_file_name": key["nwb_file_name"]}
                & {'artifact_removed_interval_list_name LIKE "%100_prop%"'}
                & {"sorter": "mountainsort4"}
            ).fetch("sort_group_id")
        )

        # 1st session
        for interval in interval_list:
            if interval[0] == "0":
                if key["sort_interval_name"].split("_")[0] in interval:
                    int_number = np.int(interval.split("_")[0])
                    print(int_number)
                    first_sess = "pos " + str(int_number - 1) + " valid times"
                # 2nd session
                elif key["sort_interval_name"].split("_")[1] in interval:
                    int_number = np.int(interval.split("_")[0])
                    print(int_number)
                    second_sess = "pos " + str(int_number - 1) + " valid times"
            elif (
                key["nwb_file_name"] == "arthur20220404_.nwb"
                and key["sort_interval_name"] == "r2_r7"
            ):
                second_sess = "pos 9 valid times"
                # print("custom second sess name", second_sess)
            elif (
                key["nwb_file_name"] == "arthur20220402_.nwb"
                and key["sort_interval_name"] == "r3_r7"
            ):
                second_sess = "pos 9 valid times"
                # print("custom second sess name", second_sess)
            elif (
                key["nwb_file_name"] == "arthur20220319_.nwb"
                and key["sort_interval_name"] == "r2_r2"
            ):
                first_sess = "pos 1 valid times"
                second_sess = "pos 3 valid times"
                # print("custom second sess name", second_sess)
            elif (
                key["nwb_file_name"] == "arthur20220319_.nwb"
                and key["sort_interval_name"] == "r2_r3"
            ):
                first_sess = "pos 3 valid times"
                second_sess = "pos 5 valid times"
                # print("custom second sess name", second_sess)

        print("tet list", tet_list)
        # print('tet list new',tet_list_new)
        print("pos intervals", first_sess, second_sess)

        spike_time_data = dict()

        # need to get spike midpoint first
        high_spike_counter = 0
        for tetrode in tet_list:
            if high_spike_counter == 0:
                #         if spike_count > 10000 and time_counter == 0:
                # first_spike = (CuratedSpikeSorting & {'nwb_file_name':'pippin20210524_.nwb'} & {'sort_group_id':sort_group}
                #      & {'sort_interval_name':'r1_r2'}).fetch_nwb()[0]['units']['spike_times'][cluster][0]
                # last_spike = (CuratedSpikeSorting & {'nwb_file_name':'pippin20210524_.nwb'} & {'sort_group_id':sort_group}
                #      & {'sort_interval_name':'r1_r2'}).fetch_nwb()[0]['units']['spike_times'][cluster][-1]
                # halfway_point = first_spike + (last_spike - first_spike)/2
                # print('total time (hours)',(last_spike - first_spike)/60/60)
                # time_counter += 1
                valid_unit_ids = (
                    CuratedSpikeSorting.Unit
                    & {"nwb_file_name": key["nwb_file_name"]}
                    & {"sort_interval_name": key["sort_interval_name"]}
                    & {'artifact_removed_interval_list_name LIKE "%100_prop%"'}
                    & {"sorter": "mountainsort4"}
                    & {"curation_id": 1}
                    & {"sort_group_id": tetrode}
                ).fetch("unit_id")

                try:
                    tetrode_spike_times = (
                        CuratedSpikeSorting
                        & {"nwb_file_name": key["nwb_file_name"]}
                        & {"sort_interval_name": key["sort_interval_name"]}
                        & {
                            'artifact_removed_interval_list_name LIKE "%100_prop%"'
                        }
                        & {"sorter": "mountainsort4"}
                        & {"curation_id": 1}
                        & {"sort_group_id": tetrode}
                    ).fetch_nwb()[0]["units"]["spike_times"]

                    for cluster in valid_unit_ids:
                        if (
                            tetrode_spike_times[cluster].shape[0] > 10000
                            and high_spike_counter == 0
                        ):
                            start_time_spike = tetrode_spike_times[cluster][0]
                            end_time_spike = tetrode_spike_times[cluster][-1]
                            mid_time_spike = start_time_spike + (
                                (end_time_spike - start_time_spike) / 2
                            )
                            print("cluster for timing", tetrode, cluster)
                            print(
                                "run1 + run2 time",
                                (end_time_spike - start_time_spike) / 60 / 60,
                            )
                            high_spike_counter += 1
                except KeyError as e:
                    pass

        for tetrode in tet_list:
            # print('tetrode:',sort_group)
            spike_time_data[tetrode] = {}

            # try:
            obj_string = (
                CuratedSpikeSorting
                & {"nwb_file_name": key["nwb_file_name"]}
                & {"sort_interval_name": key["sort_interval_name"]}
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
                    & {"sort_interval_name": key["sort_interval_name"]}
                    & {'artifact_removed_interval_list_name LIKE "%100_prop%"'}
                    & {"sorter": "mountainsort4"}
                    & {"sort_group_id": tetrode}
                ).fetch("curation_id")

                if 3 in curation_versions:
                    print("manually curated tetrode", tetrode)
                    if (
                        len(
                            (
                                CuratedSpikeSorting
                                & {"nwb_file_name": key["nwb_file_name"]}
                                & {
                                    "sort_interval_name": key[
                                        "sort_interval_name"
                                    ]
                                }
                                & {
                                    'artifact_removed_interval_list_name LIKE "%100_prop%"'
                                }
                                & {"sorter": "mountainsort4"}
                                & {"sort_group_id": tetrode}
                                & {"curation_id": 3}
                            ).fetch("units_object_id")[0]
                        )
                        > 0
                    ):
                        tetrode_spike_times = (
                            CuratedSpikeSorting
                            & {"nwb_file_name": key["nwb_file_name"]}
                            & {"sort_interval_name": key["sort_interval_name"]}
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
                            & {"sort_interval_name": key["sort_interval_name"]}
                            & {
                                'artifact_removed_interval_list_name LIKE "%100_prop%"'
                            }
                            & {"sorter": "mountainsort4"}
                            & {"curation_id": 3}
                            & {"sort_group_id": tetrode}
                        ).fetch("unit_id")
                        curation_id = 3
                    else:
                        valid_unit_ids = []
                        print("man curated tetrode, no units")

                elif 1 in curation_versions:
                    # print("regular tetrode", tetrode)
                    tetrode_spike_times = (
                        CuratedSpikeSorting
                        & {"nwb_file_name": key["nwb_file_name"]}
                        & {"sort_interval_name": key["sort_interval_name"]}
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
                        & {"sort_interval_name": key["sort_interval_name"]}
                        & {
                            'artifact_removed_interval_list_name LIKE "%100_prop%"'
                        }
                        & {"sorter": "mountainsort4"}
                        & {"curation_id": 1}
                        & {"sort_group_id": tetrode}
                    ).fetch("unit_id")
                    curation_id = 1

                # print(valid_unit_ids)

                for cluster in valid_unit_ids:
                    spike_time_data[tetrode][cluster] = tetrode_spike_times[
                        cluster
                    ]
                    # print(tetrode, cluster)

                    # remove MUA units
                    cluster_label = (
                        CuratedSpikeSorting.Unit
                        & {"nwb_file_name": key["nwb_file_name"]}
                        & {"sort_interval_name": key["sort_interval_name"]}
                        & {
                            'artifact_removed_interval_list_name LIKE "%100_prop%"'
                        }
                        & {"sorter": "mountainsort4"}
                        & {"curation_id": curation_id}
                        & {"sort_group_id": tetrode}
                        & {"unit_id": cluster}
                    ).fetch("label")[0]
                    # if cluster_label == "mua":
                    #    pass
                    #    # print("mua")
                    if cluster_label == "" or cluster_label == "mua":
                        # check for interneuron and min spike count
                        # good clusters: find histogram near reward times

                        # make start time depend on parameter name:
                        if key["tetrode_stability_param_name"] == "half_time":
                            # use different method for start and end times
                            start_time_1st = start_time_spike
                            end_time_1st = mid_time_spike

                        else:
                            start_time_1st = (
                                IntervalList
                                & {"nwb_file_name": key["nwb_file_name"]}
                                & {"interval_list_name": first_sess}
                            ).fetch("valid_times")[0][0][0]
                            end_time_1st = (
                                IntervalList
                                & {"nwb_file_name": key["nwb_file_name"]}
                                & {"interval_list_name": first_sess}
                            ).fetch("valid_times")[0][0][1]
                        session_duration_1st = end_time_1st - start_time_1st
                        session_spikes_1st = tetrode_spike_times[cluster][
                            (tetrode_spike_times[cluster] > start_time_1st)
                            & (tetrode_spike_times[cluster] < end_time_1st)
                        ]
                        # NOTE: for pippin we need a different way of calculating start and end times
                        # lets just look at an MUA unit and take midpoint of first and last spike time
                        if key["tetrode_stability_param_name"] == "half_time":
                            # use different method for start and end times
                            start_time_2nd = mid_time_spike
                            end_time_2nd = end_time_spike

                        else:
                            start_time_2nd = (
                                IntervalList
                                & {"nwb_file_name": key["nwb_file_name"]}
                                & {"interval_list_name": second_sess}
                            ).fetch("valid_times")[0][0][0]
                            end_time_2nd = (
                                IntervalList
                                & {"nwb_file_name": key["nwb_file_name"]}
                                & {"interval_list_name": second_sess}
                            ).fetch("valid_times")[0][0][1]
                        session_duration_2nd = end_time_2nd - start_time_2nd
                        session_spikes_2nd = tetrode_spike_times[cluster][
                            (tetrode_spike_times[cluster] > start_time_2nd)
                            & (tetrode_spike_times[cluster] < end_time_2nd)
                        ]

                        if (
                            session_spikes_1st.shape[0]
                            + session_spikes_2nd.shape[0]
                            > 0
                        ):
                            fraction_1st = session_spikes_1st.shape[0] / (
                                session_spikes_1st.shape[0]
                                + session_spikes_2nd.shape[0]
                            )
                            spike_imbalance_list.append(fraction_1st)
                        else:
                            print("no spikes", tetrode, cluster)

                        # print(tetrode,cluster,"total spikes",session_spikes_1st.shape[0] + session_spikes_2nd.shape[0],)

                    else:
                        print(cluster_label)

            else:
                print("no units on tetrode", tetrode)

        key["first_sess_spikes"] = session_spikes_1st.shape[0]
        key["second_sess_spikes"] = session_spikes_2nd.shape[0]
        key["fraction_first_spikes"] = spike_imbalance_list
        self.insert1(key)
