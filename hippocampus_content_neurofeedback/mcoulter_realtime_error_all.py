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

schema = dj.schema("mcoulter_realtime_error_all")


# parameters
@schema
class RealtimeErrorAllParameters(dj.Manual):
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
        self.insert1(["default", realtime_error_parameters], skip_duplicates=True)


# i dont think we need samplecount or statescriptfile in this list
# select parameters and cluster
@schema
class RealtimeErrorAllSelection(dj.Manual):
    definition = """
    -> RealtimeFilename
    -> RealtimeErrorAllParameters
    ---
    """


# this is the computed table - basically it has the results of the analysis
@schema
class RealtimeErrorAll(dj.Computed):
    definition = """
    -> RealtimeErrorAllSelection
    ---
    error_table : blob

    """

    def make(self, key):
        print(f"Computing realtime error for: {key}")

        if key["nwb_file_name"].split("2")[0] == key["realtime_filename"].split("_")[3]:
            print("names match")
        elif (
            key["nwb_file_name"].split("2")[0] == key["realtime_filename"].split("_")[2]
        ):
            print("names match")
        else:
            raise ValueError("rat names dont match")

        # these are all in the paramters set now
        error_parameters = (
            RealtimeErrorAllParameters
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

        # only do analysis if there is taskstate2
        # should we also only do it if there is task3?
        if decoder_data[decoder_data["taskState"] == 2].shape[0] > 0:
            task2_start_ts = decoder_data[decoder_data["taskState"] == 2][0:1][
                "bin_timestamp"
            ].values[0]

            # this cell makes intervals for each position bin from occupancy_data
            occupancy_data_1 = occupancy_data.copy()
            position_intervals = (
                occupancy_data_1.reset_index()
                .groupby(
                    (
                        occupancy_data_1["linear_pos"]
                        != occupancy_data_1["linear_pos"].shift()
                    ).cumsum(),
                    as_index=False,
                )
                .agg(
                    {
                        "linear_pos": [("position", "first")],
                        "index": [("first", "first"), ("last", "last")],
                    }
                )
            )
            position_intervals["start_ts"] = occupancy_data_1.iloc[
                position_intervals[("index", "first")].values.tolist()
            ]["bin_timestamp"].values
            position_intervals["end_ts"] = occupancy_data_1.iloc[
                position_intervals[("index", "last")].values.tolist()
            ]["bin_timestamp"].values

            # try to only include the outbound run - decoding is generally better

            # loop through visits to arm1 and find interval for whole run down arm
            # NOTE: does not include last run
            intervals_24 = position_intervals[
                position_intervals[("linear_pos", "position")] == 24
            ]
            arm1_runs = np.zeros((intervals_24.shape[0], 2))
            arm1_in_runs = np.zeros((intervals_24.shape[0], 2))
            arm1_out_runs = np.zeros((intervals_24.shape[0], 2))

            for arm_end in np.arange(intervals_24.shape[0] - 1):
                leave_arm_end_index = intervals_24.iloc[[arm_end]][
                    ("index", "last")
                ].values[0]
                if arm_end == intervals_24.shape[0] - 1:
                    # arm1_runs[arm_end,1] = occupancy_data_1.iloc[[leave_arm_end_index]]['bin_timestamp'].values
                    # print('last one')
                    pass
                else:
                    for i in np.arange(occupancy_data.shape[0]):
                        # print(occupancy_data_1.iloc[[leave_arm_end_index+i]]['linear_pos'].values)
                        if (leave_arm_end_index + i) < occupancy_data_1.shape[0]:
                            if (
                                occupancy_data_1.iloc[[leave_arm_end_index + i]][
                                    "linear_pos"
                                ].values
                            ) == center_well_bin:
                                # print('arm end visit',arm_end,i,'center well',intervals_24.shape[0])

                                arm1_runs[arm_end, 1] = occupancy_data_1.iloc[
                                    [leave_arm_end_index + i]
                                ]["bin_timestamp"].values
                                arm1_in_runs[arm_end, 1] = occupancy_data_1.iloc[
                                    [leave_arm_end_index + i]
                                ]["bin_timestamp"].values
                                arm1_in_runs[arm_end, 0] = np.mean(
                                    [
                                        intervals_24["end_ts"].values[arm_end],
                                        intervals_24["start_ts"].values[arm_end],
                                    ]
                                )
                                break
                        else:
                            print("last position not center")
                            break

            for arm_end in np.arange(intervals_24.shape[0]):
                leave_arm_end_index = intervals_24.iloc[[arm_end]][
                    ("index", "first")
                ].values[0]
                if arm_end == intervals_24.shape[0] - 1:
                    pass
                else:
                    for i in np.arange(occupancy_data.shape[0]):
                        # print(occupancy_data_1.iloc[[leave_arm_end_index+i]]['linear_pos'].values)
                        if (leave_arm_end_index + i) < occupancy_data_1.shape[0]:
                            if (
                                occupancy_data_1.iloc[[leave_arm_end_index - i]][
                                    "linear_pos"
                                ].values
                            ) == center_well_bin:
                                # print('arm end visit',arm_end,i,'center well')

                                arm1_runs[arm_end, 0] = occupancy_data_1.iloc[
                                    [leave_arm_end_index - i]
                                ]["bin_timestamp"].values
                                arm1_out_runs[arm_end, 0] = occupancy_data_1.iloc[
                                    [leave_arm_end_index - i]
                                ]["bin_timestamp"].values
                                arm1_out_runs[arm_end, 1] = np.mean(
                                    [
                                        intervals_24["end_ts"].values[arm_end],
                                        intervals_24["start_ts"].values[arm_end],
                                    ]
                                )
                                break
                        else:
                            print("last position not center")
                            break

            arm1_runs
            arm1_runs_pandas = pd.DataFrame(arm1_runs, columns=["start", "end"])
            arm1_in_runs_pandas = pd.DataFrame(arm1_in_runs, columns=["start", "end"])
            arm1_out_runs_pandas = pd.DataFrame(arm1_out_runs, columns=["start", "end"])
            print('arm1 runs',arm1_out_runs_pandas.shape)

            # loop through visits to arm2 and find interval for whole run down arm
            # NOTE: does not include last run during task3
            intervals_40 = position_intervals[
                position_intervals[("linear_pos", "position")] == 40
            ]
            arm2_runs = np.zeros((intervals_40.shape[0], 2))
            arm2_in_runs = np.zeros((intervals_40.shape[0], 2))
            arm2_out_runs = np.zeros((intervals_40.shape[0], 2))
            

            for arm_end in np.arange(intervals_40.shape[0]):
                leave_arm_end_index = intervals_40.iloc[[arm_end]][
                    ("index", "last")
                ].values[0]
                if arm_end == intervals_40.shape[0] - 1:
                    pass
                else:
                    for i in np.arange(occupancy_data.shape[0]):
                        # print(occupancy_data_1.iloc[[leave_arm_end_index+i]]['linear_pos'].values)
                        if (leave_arm_end_index + i) < occupancy_data_1.shape[0]:
                            if (
                                occupancy_data_1.iloc[[leave_arm_end_index + i]][
                                    "linear_pos"
                                ].values
                            ) == center_well_bin:
                                # print('arm end visit',arm_end,i,'center well')

                                arm2_runs[arm_end, 1] = occupancy_data_1.iloc[
                                    [leave_arm_end_index + i]
                                ]["bin_timestamp"].values
                                arm2_in_runs[arm_end, 1] = occupancy_data_1.iloc[
                                    [leave_arm_end_index + i]
                                ]["bin_timestamp"].values
                                arm2_in_runs[arm_end, 0] = np.mean(
                                    [
                                        intervals_40["end_ts"].values[arm_end],
                                        intervals_40["start_ts"].values[arm_end],
                                    ]
                                )
                                break
                        else:
                            print("not center at end")
                            break

            for arm_end in np.arange(intervals_40.shape[0]):
                leave_arm_end_index = intervals_40.iloc[[arm_end]][
                    ("index", "first")
                ].values[0]
                if arm_end == intervals_40.shape[0] - 1:
                    pass
                else:
                    for i in np.arange(occupancy_data.shape[0]):
                        # print(occupancy_data_1.iloc[[leave_arm_end_index+i]]['linear_pos'].values)
                        if (leave_arm_end_index + i) < occupancy_data_1.shape[0]:
                            if (
                                occupancy_data_1.iloc[[leave_arm_end_index - i]][
                                    "linear_pos"
                                ].values
                            ) == center_well_bin:
                                # print('arm end visit',arm_end,i,'center well')

                                arm2_runs[arm_end, 0] = occupancy_data_1.iloc[
                                    [leave_arm_end_index - i]
                                ]["bin_timestamp"].values
                                arm2_out_runs[arm_end, 0] = occupancy_data_1.iloc[
                                    [leave_arm_end_index - i]
                                ]["bin_timestamp"].values
                                arm2_out_runs[arm_end, 1] = np.mean(
                                    [
                                        intervals_40["end_ts"].values[arm_end],
                                        intervals_40["start_ts"].values[arm_end],
                                    ]
                                )
                                break
                        else:
                            print("not center at end")
                            break

            arm2_runs
            arm2_runs_pandas = pd.DataFrame(arm2_runs, columns=["start", "end"])
            arm2_in_runs_pandas = pd.DataFrame(arm2_in_runs, columns=["start", "end"])
            arm2_out_runs_pandas = pd.DataFrame(arm2_out_runs, columns=["start", "end"])
            print('arm2 runs',arm2_out_runs_pandas.shape)

            # full arm runs
            if in_out == 1:
                arm1_last_three = arm1_runs_pandas[
                    (arm1_runs_pandas["start"] < task2_start_ts)
                    & (arm1_runs_pandas["start"] > 1)
                ][-number_of_runs:]
                arm2_last_three = arm2_runs_pandas[
                    (arm2_runs_pandas["start"] < task2_start_ts)
                    & (arm2_runs_pandas["start"] > 1)
                ][-number_of_runs:]
            # outbound arm runs
            elif in_out == 2:
                arm1_last_three = arm1_out_runs_pandas[
                    (arm1_out_runs_pandas["start"] < task2_start_ts)
                    & (arm1_out_runs_pandas["start"] > 1)
                ][-number_of_runs:]
                arm2_last_three = arm2_out_runs_pandas[
                    (arm2_out_runs_pandas["start"] < task2_start_ts)
                    & (arm2_out_runs_pandas["start"] > 1)
                ][-number_of_runs:]
                # inbound arm runs
            elif in_out == 3:
                arm1_last_three = arm1_in_runs_pandas[
                    (arm1_in_runs_pandas["start"] < task2_start_ts)
                    & (arm1_in_runs_pandas["start"] > 1)
                ][-number_of_runs:]
                arm2_last_three = arm2_in_runs_pandas[
                    (arm2_in_runs_pandas["start"] < task2_start_ts)
                    & (arm2_in_runs_pandas["start"] > 1)
                ][-number_of_runs:]

            # test with first 3 visits
            # arm1_last_three = arm1_runs_pandas[(arm1_runs_pandas['start']<task2_start_ts)&(arm1_runs_pandas['start']>1)][0:3]
            # arm2_last_three = arm2_runs_pandas[(arm2_runs_pandas['start']<task2_start_ts)&(arm2_runs_pandas['start']>1)][0:3]

            # all different
            # choose earliest time

            print(arm1_last_three.shape, arm2_last_three.shape)

            try:
                if arm1_last_three["start"].values[0] < arm2_last_three["start"].values[0]:
                    last_3_start_time = arm1_last_three["start"].values[0].copy()
                elif (
                    arm1_last_three["start"].values[0] > arm2_last_three["start"].values[0]
                ):
                    last_3_start_time = arm2_last_three["start"].values[0].copy()

                arm_run_decoder = decoder_data[
                    (decoder_data["bin_timestamp"] > last_3_start_time)
                    & (decoder_data["bin_timestamp"] < task2_start_ts)
                    & (decoder_data["velocity"] > velocity_filter)
                ]
                arm_run_decoder["post_max"] = (
                    arm_run_decoder.iloc[:, 27:68]
                    .idxmax(axis=1)
                    .str.slice(1, 3, 1)
                    .astype("int16")
                )

                # need to remove time bins with 0 spikes - no posterior
                arm_run_decoder_spikes = arm_run_decoder[arm_run_decoder["spike_count"] > 0]
                print("bins for decoder quality", arm_run_decoder_spikes.shape)

                # need to break into 9 tables for accurate error distance measure
                ## ARM 1
                arm_run_decoder_spikes_1_to_box = arm_run_decoder_spikes[
                    (arm_run_decoder_spikes["real_pos"] > 10)
                    & (arm_run_decoder_spikes["real_pos"] < 26)
                    & (arm_run_decoder_spikes["post_max"] < 10)
                ]
                accurate_error_1_to_box = np.abs(
                    (
                        arm_run_decoder_spikes_1_to_box["real_pos"]
                        - (arm_run_decoder_spikes_1_to_box["post_max"] + 4)
                    )
                )

                arm_run_decoder_spikes_1_to_2 = arm_run_decoder_spikes[
                    (arm_run_decoder_spikes["real_pos"] > 10)
                    & (arm_run_decoder_spikes["real_pos"] < 26)
                    & (arm_run_decoder_spikes["post_max"] > 26)
                ]

                accurate_error_1_to_2 = np.abs(
                    (
                        (arm_run_decoder_spikes_1_to_2["real_pos"] - 13)
                        - (arm_run_decoder_spikes_1_to_2["post_max"] - 29) * -1
                    )
                )

                arm_run_decoder_spikes_1_to_1 = arm_run_decoder_spikes[
                    (arm_run_decoder_spikes["real_pos"] > 10)
                    & (arm_run_decoder_spikes["real_pos"] < 26)
                    & (arm_run_decoder_spikes["post_max"] > 10)
                    & (arm_run_decoder_spikes["post_max"] < 26)
                ]
                accurate_error_1_to_1 = np.abs(
                    (
                        arm_run_decoder_spikes_1_to_1["real_pos"]
                        - (arm_run_decoder_spikes_1_to_1["post_max"] + 0)
                    )
                )

                ## ARM 2
                arm_run_decoder_spikes_2_to_box = arm_run_decoder_spikes[
                    (arm_run_decoder_spikes["real_pos"] > 26)
                    & (arm_run_decoder_spikes["post_max"] < 10)
                ]
                accurate_error_2_to_box = np.abs(
                    (
                        arm_run_decoder_spikes_2_to_box["real_pos"]
                        - (arm_run_decoder_spikes_2_to_box["post_max"] + 4 + 16)
                    )
                )

                arm_run_decoder_spikes_2_to_2 = arm_run_decoder_spikes[
                    (arm_run_decoder_spikes["real_pos"] > 26)
                    & (arm_run_decoder_spikes["post_max"] > 26)
                ]
                accurate_error_2_to_2 = np.abs(
                    (
                        arm_run_decoder_spikes_2_to_2["real_pos"]
                        - (arm_run_decoder_spikes_2_to_2["post_max"] + 0)
                    )
                )

                arm_run_decoder_spikes_2_to_1 = arm_run_decoder_spikes[
                    (arm_run_decoder_spikes["real_pos"] > 26)
                    & (arm_run_decoder_spikes["post_max"] > 10)
                    & (arm_run_decoder_spikes["post_max"] < 26)
                ]
                accurate_error_2_to_1 = np.abs(
                    (
                        (arm_run_decoder_spikes_2_to_1["real_pos"] - 29)
                        - (arm_run_decoder_spikes_2_to_1["post_max"] - 13) * -1
                    )
                )

                ## BOX
                arm_run_decoder_spikes_box_to_2 = arm_run_decoder_spikes[
                    (arm_run_decoder_spikes["real_pos"] < 10)
                    & (arm_run_decoder_spikes["post_max"] > 26)
                ]
                accurate_error_box_to_2 = np.abs(
                    (
                        (arm_run_decoder_spikes_box_to_2["real_pos"] + 4 + 16)
                        - (arm_run_decoder_spikes_box_to_2["post_max"])
                    )
                )

                arm_run_decoder_spikes_box_to_box = arm_run_decoder_spikes[
                    (arm_run_decoder_spikes["real_pos"] < 10)
                    & (arm_run_decoder_spikes["post_max"] < 10)
                ]
                accurate_error_box_to_box = np.abs(
                    (
                        arm_run_decoder_spikes_box_to_box["real_pos"]
                        - (arm_run_decoder_spikes_box_to_box["post_max"] + 0)
                    )
                )

                arm_run_decoder_spikes_box_to_1 = arm_run_decoder_spikes[
                    (arm_run_decoder_spikes["real_pos"] < 10)
                    & (arm_run_decoder_spikes["post_max"] > 10)
                    & (arm_run_decoder_spikes["post_max"] < 26)
                ]
                accurate_error_box_to_1 = np.abs(
                    (
                        (arm_run_decoder_spikes_box_to_1["real_pos"] + 4)
                        - (arm_run_decoder_spikes_box_to_1["post_max"])
                    )
                )

                # combine all errors
                all_decoding_error = pd.concat(
                    [
                        accurate_error_1_to_box * 5,
                        accurate_error_1_to_2 * 5,
                        accurate_error_1_to_1 * 5,
                        accurate_error_2_to_box * 5,
                        accurate_error_2_to_2 * 5,
                        accurate_error_2_to_1 * 5,
                        accurate_error_box_to_box * 5,
                        accurate_error_box_to_2 * 5,
                        accurate_error_box_to_1 * 5,
                    ]
                )
                print(all_decoding_error.shape)
            except IndexError as e:
                print('arm run dataframe empty')
                all_decoding_error = [0,0,0]
                
        key["error_table"] = np.array(all_decoding_error)
        self.insert1(key)
