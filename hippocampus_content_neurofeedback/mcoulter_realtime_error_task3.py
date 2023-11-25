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

schema = dj.schema("mcoulter_realtime_error")


# parameters
@schema
class RealtimeError3rdParameters(dj.Manual):
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
class RealtimeError3rdSelection(dj.Manual):
    definition = """
    -> RealtimeFilename
    -> RealtimeError3rdParameters
    ---
    """


# this is the computed table - basically it has the results of the analysis
@schema
class RealtimeError3rd(dj.Computed):
    definition = """
    -> RealtimeError3rdSelection
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
            RealtimeError3rdParameters
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

        # only do analysis if there is taskstate3
        # should we also only do it if there is task3?
        if decoder_data[decoder_data["taskState"] == 3].shape[0] > 0:
            task3_start_ts = decoder_data[decoder_data["taskState"] == 3][0:1][
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
            # NOTE: does not include last run - need to add now
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
                # if arm_end == intervals_24.shape[0] - 1:
                #    # arm1_runs[arm_end,1] = occupancy_data_1.iloc[[leave_arm_end_index]]['bin_timestamp'].values
                #    # print('last one')
                #    pass
                # else:
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
                # if arm_end == intervals_24.shape[0] - 1:
                #    pass
                # else:
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

            # this is where problems come in with the last run
            # try removing runs where start value is >= end value
            for row in np.arange(arm1_out_runs.shape[0]):
                if arm1_out_runs[row][0] >= arm1_out_runs[row][1]:
                    print("bad arm1 run")
                    arm1_out_runs[row] = [0, 0]

            arm1_runs_pandas = pd.DataFrame(arm1_runs, columns=["start", "end"])
            arm1_in_runs_pandas = pd.DataFrame(arm1_in_runs, columns=["start", "end"])
            arm1_out_runs_pandas = pd.DataFrame(arm1_out_runs, columns=["start", "end"])

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
                # if arm_end == intervals_40.shape[0] - 1:
                #    pass
                # else:
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
                # if arm_end == intervals_40.shape[0] - 1:
                #    pass
                # else:
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

            # this is where problems come in with the last run
            # try removing runs where start value is >= end value
            for row in np.arange(arm2_out_runs.shape[0]):
                if arm2_out_runs[row][0] >= arm2_out_runs[row][1]:
                    print("bad arm2 run")
                    arm2_out_runs[row] = [0, 0]

            arm2_runs_pandas = pd.DataFrame(arm2_runs, columns=["start", "end"])
            arm2_in_runs_pandas = pd.DataFrame(arm2_in_runs, columns=["start", "end"])
            arm2_out_runs_pandas = pd.DataFrame(arm2_out_runs, columns=["start", "end"])

            # updated for task3
            # full arm runs
            if in_out == 1:
                arm1_last_three = arm1_runs_pandas[
                    (arm1_runs_pandas["start"] > task3_start_ts)
                ]
                arm2_last_three = arm2_runs_pandas[
                    (arm2_runs_pandas["start"] > task3_start_ts)
                ]
                print(
                    "task 3 runs (full). arm 1",
                    arm1_last_three.shape[0],
                    "arm 2",
                    arm2_last_three.shape[0],
                )

            # outbound arm runs
            elif in_out == 2:
                arm1_last_three = arm1_out_runs_pandas[
                    (arm1_out_runs_pandas["start"] > task3_start_ts)
                ]
                arm2_last_three = arm2_out_runs_pandas[
                    (arm2_out_runs_pandas["start"] > task3_start_ts)
                ]
                print(
                    "task 3 runs (outbound). arm 1",
                    arm1_last_three.shape[0],
                    "arm 2",
                    arm2_last_three.shape[0],
                )

            # inbound arm runs
            elif in_out == 3:
                arm1_last_three = arm1_in_runs_pandas[
                    (arm1_in_runs_pandas["start"] > task3_start_ts)
                ]
                arm2_last_three = arm2_in_runs_pandas[
                    (arm2_in_runs_pandas["start"] > task3_start_ts)
                ]
                print(
                    "task 3 runs (inbound). arm 1",
                    arm1_last_three.shape[0],
                    "arm 2",
                    arm2_last_three.shape[0],
                )

            # test with first 3 visits
            # arm1_last_three = arm1_runs_pandas[(arm1_runs_pandas['start']<task2_start_ts)&(arm1_runs_pandas['start']>1)][0:3]
            # arm2_last_three = arm2_runs_pandas[(arm2_runs_pandas['start']<task2_start_ts)&(arm2_runs_pandas['start']>1)][0:3]

            # check that arms exist
            if arm1_last_three.shape[0] > 0 and arm2_last_three.shape[0] > 0:
                if arm1_last_three.shape[0] > 2:
                    # first get average timebin for position 23 and 39
                    bin_23_1 = decoder_data[
                        (
                            decoder_data["bin_timestamp"]
                            > arm1_last_three["start"].values[0]
                        )
                        & (
                            decoder_data["bin_timestamp"]
                            < arm1_last_three["end"].values[0]
                        )
                        & (decoder_data["velocity"] > velocity_filter)
                        & (decoder_data["real_pos"] == 23)
                    ].shape[0]
                    bin_23_2 = decoder_data[
                        (
                            decoder_data["bin_timestamp"]
                            > arm1_last_three["start"].values[1]
                        )
                        & (
                            decoder_data["bin_timestamp"]
                            < arm1_last_three["end"].values[1]
                        )
                        & (decoder_data["velocity"] > velocity_filter)
                        & (decoder_data["real_pos"] == 23)
                    ].shape[0]
                    bin_23_3 = decoder_data[
                        (
                            decoder_data["bin_timestamp"]
                            > arm1_last_three["start"].values[2]
                        )
                        & (
                            decoder_data["bin_timestamp"]
                            < arm1_last_three["end"].values[2]
                        )
                        & (decoder_data["velocity"] > velocity_filter)
                        & (decoder_data["real_pos"] == 23)
                    ].shape[0]

                    bin_23_avg = int(np.average([bin_23_1, bin_23_2, bin_23_3]) / 2)
                    # statistics.mean([])
                    print("position 23 average time bins (3 runs)", bin_23_avg)

                elif arm1_last_three.shape[0] == 2:
                    # first get average timebin for position 23 and 39
                    bin_23_1 = decoder_data[
                        (
                            decoder_data["bin_timestamp"]
                            > arm1_last_three["start"].values[0]
                        )
                        & (
                            decoder_data["bin_timestamp"]
                            < arm1_last_three["end"].values[0]
                        )
                        & (decoder_data["velocity"] > velocity_filter)
                        & (decoder_data["real_pos"] == 23)
                    ].shape[0]
                    bin_23_2 = decoder_data[
                        (
                            decoder_data["bin_timestamp"]
                            > arm1_last_three["start"].values[1]
                        )
                        & (
                            decoder_data["bin_timestamp"]
                            < arm1_last_three["end"].values[1]
                        )
                        & (decoder_data["velocity"] > velocity_filter)
                        & (decoder_data["real_pos"] == 23)
                    ].shape[0]

                    bin_23_avg = int(np.average([bin_23_1, bin_23_2]) / 2)
                    # statistics.mean([])
                    print("position 23 average time bins (2 runs)", bin_23_avg)

                elif arm1_last_three.shape[0] == 1:
                    # first get average timebin for position 23 and 39
                    bin_23_1 = decoder_data[
                        (
                            decoder_data["bin_timestamp"]
                            > arm1_last_three["start"].values[0]
                        )
                        & (
                            decoder_data["bin_timestamp"]
                            < arm1_last_three["end"].values[0]
                        )
                        & (decoder_data["velocity"] > velocity_filter)
                        & (decoder_data["real_pos"] == 23)
                    ].shape[0]

                    bin_23_avg = int(bin_23_1 / 2)
                    # statistics.mean([])
                    print("position 23 average time bins (1 run)", bin_23_avg)

                if arm2_last_three.shape[0] > 2:
                    bin_39_1 = decoder_data[
                        (
                            decoder_data["bin_timestamp"]
                            > arm2_last_three["start"].values[0]
                        )
                        & (
                            decoder_data["bin_timestamp"]
                            < arm2_last_three["end"].values[0]
                        )
                        & (decoder_data["velocity"] > velocity_filter)
                        & (decoder_data["real_pos"] == 39)
                    ].shape[0]
                    bin_39_2 = decoder_data[
                        (
                            decoder_data["bin_timestamp"]
                            > arm2_last_three["start"].values[1]
                        )
                        & (
                            decoder_data["bin_timestamp"]
                            < arm2_last_three["end"].values[1]
                        )
                        & (decoder_data["velocity"] > velocity_filter)
                        & (decoder_data["real_pos"] == 39)
                    ].shape[0]
                    bin_39_3 = decoder_data[
                        (
                            decoder_data["bin_timestamp"]
                            > arm2_last_three["start"].values[2]
                        )
                        & (
                            decoder_data["bin_timestamp"]
                            < arm2_last_three["end"].values[2]
                        )
                        & (decoder_data["velocity"] > velocity_filter)
                        & (decoder_data["real_pos"] == 39)
                    ].shape[0]

                    bin_39_avg = int(np.average([bin_39_1, bin_39_2, bin_39_3]) / 2)
                    print("position 39 average time bins (3 runs)", bin_39_avg)

                elif arm2_last_three.shape[0] == 2:
                    bin_39_1 = decoder_data[
                        (
                            decoder_data["bin_timestamp"]
                            > arm2_last_three["start"].values[0]
                        )
                        & (
                            decoder_data["bin_timestamp"]
                            < arm2_last_three["end"].values[0]
                        )
                        & (decoder_data["velocity"] > velocity_filter)
                        & (decoder_data["real_pos"] == 39)
                    ].shape[0]
                    bin_39_2 = decoder_data[
                        (
                            decoder_data["bin_timestamp"]
                            > arm2_last_three["start"].values[1]
                        )
                        & (
                            decoder_data["bin_timestamp"]
                            < arm2_last_three["end"].values[1]
                        )
                        & (decoder_data["velocity"] > velocity_filter)
                        & (decoder_data["real_pos"] == 39)
                    ].shape[0]

                    bin_39_avg = int(np.average([bin_39_1, bin_39_2]) / 2)
                    print("position 39 average time bins (2 runs)", bin_39_avg)

                elif arm2_last_three.shape[0] == 1:
                    bin_39_1 = decoder_data[
                        (
                            decoder_data["bin_timestamp"]
                            > arm2_last_three["start"].values[0]
                        )
                        & (
                            decoder_data["bin_timestamp"]
                            < arm2_last_three["end"].values[0]
                        )
                        & (decoder_data["velocity"] > velocity_filter)
                        & (decoder_data["real_pos"] == 39)
                    ].shape[0]

                    bin_39_avg = int(bin_39_1 / 2)
                    print("position 39 average time bins (1 run)", bin_39_avg)

                # arm 1 - typo in original script, this next line should be number_of_runs
                # for j in np.arange(number_of_runs):

                if arm1_last_three.shape[0] > number_of_runs:
                    arm_run_loops = number_of_runs
                else:
                    arm_run_loops = arm1_last_three.shape[0]

                for j in np.arange(arm_run_loops):
                    arm_run_decoder = decoder_data[
                        (
                            decoder_data["bin_timestamp"]
                            > arm1_last_three["start"].values[j]
                        )
                        & (
                            decoder_data["bin_timestamp"]
                            < arm1_last_three["end"].values[j]
                        )
                        & (decoder_data["velocity"] > velocity_filter)
                    ]
                    arm_run_decoder["post_max"] = (
                        arm_run_decoder.iloc[:, 27:68]
                        .idxmax(axis=1)
                        .str.slice(1, 3, 1)
                        .astype("int16")
                    )

                    # want to only use beginning and end times at the well, can use average of next to last position
                    # print(arm_run_decoder[arm_run_decoder['real_pos']==23].shape[0]) - divide this by 2
                    # ideally find this number for all 3 runs and average all
                    # then if position == 24, only do calcuation on a subset of it

                    # select columns by index: df.iloc[:, 0:3]
                    for i in np.arange(13, 25):
                        posterior_fraction_array[i - 13, 0] = i
                        new_dataframe = arm_run_decoder[
                            arm_run_decoder["real_pos"] == i
                        ]
                        # print(new_dataframe.shape)
                        # this was to calculate fraction posterior +/- 2 bins of real position

                        # error distance (from post max to actual pos when spike > 0)
                        if i == 24:
                            new_dataframe_begin = new_dataframe[:bin_23_avg]
                            new_dataframe_end = new_dataframe[-bin_23_avg:]
                            # new_dataframe_2 = pd.concat(new_dataframe_begin,new_dataframe_end)
                            new_dataframe_2 = new_dataframe_begin.append(
                                new_dataframe_end, ignore_index=True
                            )
                            new_dataframe_3 = new_dataframe_2[
                                new_dataframe_2["spike_count"] > 0
                            ]
                            # this is distance for any position bin
                            posterior_position_dist = (
                                np.average(
                                    new_dataframe_3["real_pos"]
                                    - new_dataframe_3["post_max"]
                                )
                                * 5
                            )
                            # this is to record where max is in other arm or box
                            local_bins = new_dataframe_3[
                                (new_dataframe_3["post_max"] > 10)
                                & (new_dataframe_3["post_max"] < 26)
                            ].shape[0]
                            # only do this is shape > 0
                            if new_dataframe_3.shape[0] > 0:
                                posterior_fraction_array[i - 13, j + 4] = 1 - (
                                    local_bins / new_dataframe_3.shape[0]
                                )
                                posterior_fraction_array[
                                    i - 13, j + 1
                                ] = new_dataframe_3.shape[0]
                                # posterior distance
                                # posterior_fraction_array[i-13,j+4,session] = posterior_position_dist

                            # check by plotting position
                            # plt.figure(figsize=(8,3))
                            # plt.scatter(arm_run_decoder['bin_timestamp'].values,arm_run_decoder['real_pos'].values,s=1,color='black')
                            # plt.scatter(new_dataframe_2['bin_timestamp'].values,new_dataframe_2['real_pos'].values,s=1,color='red')

                        else:
                            new_dataframe_3 = new_dataframe[
                                new_dataframe["spike_count"] > 0
                            ]
                            # for +/- 2 position bins: 36,41
                            posterior_position_dist = (
                                np.average(
                                    new_dataframe_3["real_pos"]
                                    - new_dataframe_3["post_max"]
                                )
                                * 5
                            )
                            # print('position',i,'posterior fraction',posterior_fraction_position)
                            posterior_fraction_array[
                                i - 13, j + 1
                            ] = new_dataframe_3.shape[0]
                            # posterior distance
                            # posterior_fraction_array[i-13,j+4,session] = posterior_position_dist
                            local_bins = new_dataframe_3[
                                (new_dataframe_3["post_max"] > 10)
                                & (new_dataframe_3["post_max"] < 26)
                            ].shape[0]
                            # only do this is shape > 0
                            if new_dataframe_3.shape[0] > 0:
                                posterior_fraction_array[i - 13, j + 4] = 1 - (
                                    local_bins / new_dataframe_3.shape[0]
                                )

                # arm 2
                # for j in np.arange(number_of_bins):

                if arm2_last_three.shape[0] > number_of_runs:
                    arm_run_loops = number_of_runs
                else:
                    arm_run_loops = arm2_last_three.shape[0]

                for j in np.arange(arm_run_loops):
                    arm_run_decoder = decoder_data[
                        (
                            decoder_data["bin_timestamp"]
                            > arm2_last_three["start"].values[j]
                        )
                        & (
                            decoder_data["bin_timestamp"]
                            < arm2_last_three["end"].values[j]
                        )
                        & (decoder_data["velocity"] > velocity_filter)
                    ]
                    arm_run_decoder["post_max"] = (
                        arm_run_decoder.iloc[:, 27:68]
                        .idxmax(axis=1)
                        .str.slice(1, 3, 1)
                        .astype("int16")
                    )

                    # select columns by index: df.iloc[:, 0:3]
                    for i in np.arange(29, 41):
                        posterior_fraction_array[i - 16, 0] = i
                        new_dataframe = arm_run_decoder[
                            arm_run_decoder["real_pos"] == i
                        ]
                        # print(i,new_dataframe.shape)
                        posterior_fraction_array[i - 16, j + 1] = new_dataframe.shape[0]

                        # distance from position to posterior max
                        # need to do update below
                        if i == 39:
                            new_dataframe_3 = new_dataframe[
                                new_dataframe["spike_count"] > 0
                            ]
                            # for +/- 2 position bins: 36,41
                            # here 5 is cm per position bin
                            posterior_position_dist = (
                                np.average(
                                    new_dataframe_3["real_pos"]
                                    - new_dataframe_3["post_max"]
                                )
                                * 5
                            )
                            # print('position',i,'posterior fraction',posterior_fraction_position)
                            posterior_fraction_array[
                                i - 16, j + 1
                            ] = new_dataframe_3.shape[0]
                            # posterior distance
                            # posterior_fraction_array[i-16,j+4,session] = posterior_position_dist
                            local_bins = new_dataframe_3[
                                (new_dataframe_3["post_max"] > 26)
                                & (new_dataframe_3["post_max"] < 41)
                            ].shape[0]
                            # only do this is shape > 0
                            if new_dataframe_3.shape[0] > 0:
                                posterior_fraction_array[i - 16, j + 4] = 1 - (
                                    local_bins / new_dataframe_3.shape[0]
                                )

                        elif i == 40:
                            new_dataframe_begin = new_dataframe[:bin_39_avg]
                            new_dataframe_end = new_dataframe[-bin_39_avg:]
                            # new_dataframe_2 = pd.concat(new_dataframe_begin,new_dataframe_end)
                            new_dataframe_2 = new_dataframe_begin.append(
                                new_dataframe_end, ignore_index=True
                            )
                            new_dataframe_3 = new_dataframe_2[
                                new_dataframe_2["spike_count"] > 0
                            ]
                            posterior_position_dist = (
                                np.average(
                                    new_dataframe_3["real_pos"]
                                    - new_dataframe_3["post_max"]
                                )
                                * 5
                            )
                            posterior_fraction_array[
                                i - 16, j + 1
                            ] = new_dataframe_3.shape[0]
                            # postior distance
                            # posterior_fraction_array[i-16,j+4,session] = posterior_position_dist
                            local_bins = new_dataframe_3[
                                (new_dataframe_3["post_max"] > 26)
                                & (new_dataframe_3["post_max"] < 41)
                            ].shape[0]
                            # only do this is shape > 0
                            if new_dataframe_3.shape[0] > 0:
                                posterior_fraction_array[i - 16, j + 4] = 1 - (
                                    local_bins / new_dataframe_3.shape[0]
                                )

                            # check by plotting position
                            # plt.figure(figsize=(8,3))
                            # plt.scatter(arm_run_decoder['bin_timestamp'].values,arm_run_decoder['real_pos'].values,s=1,color='black')
                            # plt.scatter(new_dataframe_2['bin_timestamp'].values,new_dataframe_2['real_pos'].values,s=1,color='red')

                        else:
                            new_dataframe_3 = new_dataframe[
                                new_dataframe["spike_count"] > 0
                            ]
                            # for +/- 2 position bins: 36,41
                            posterior_position_dist = (
                                np.average(
                                    new_dataframe_3["real_pos"]
                                    - new_dataframe_3["post_max"]
                                )
                                * 5
                            )
                            # print('position',i,'posterior fraction',posterior_fraction_position)
                            posterior_fraction_array[
                                i - 16, j + 1
                            ] = new_dataframe_3.shape[0]
                            # posterior distance
                            # posterior_fraction_array[i-16,j+4,session] = posterior_position_dist
                            local_bins = new_dataframe_3[
                                (new_dataframe_3["post_max"] > 26)
                                & (new_dataframe_3["post_max"] < 41)
                            ].shape[0]
                            # only do this is shape > 0
                            if new_dataframe_3.shape[0] > 0:
                                posterior_fraction_array[i - 16, j + 4] = 1 - (
                                    local_bins / new_dataframe_3.shape[0]
                                )

                        # print('position',i,'posterior fraction',posterior_fraction_position)
                        # posterior_fraction_array[i-16,j+4,session] = posterior_fraction_position
        else:
            print("no task3")

        key["error_table"] = posterior_fraction_array
        self.insert1(key)
