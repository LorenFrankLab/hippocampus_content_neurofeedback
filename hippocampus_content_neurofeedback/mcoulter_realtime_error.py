# required packages:
import datajoint as dj
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
#from scipy.ndimage import gaussian_filter
#from spyglass.decoding.core import (
#    convert_epoch_interval_name_to_position_interval_name,
#    convert_valid_times_to_slice, get_valid_ephys_position_times_by_epoch)

from spyglass.common import IntervalPositionInfo, IntervalList, StateScriptFile, DIOEvents, SampleCount
from spyglass.common.dj_helper_fn import fetch_nwb
#from replay_trajectory_classification.environments import Environment
#from spyglass.common.common_position import TrackGraph
#from spyglass.common.common_nwbfile import AnalysisNwbfile
#from spyglass.decoding.clusterless import ClusterlessClassifierParameters

import pprint

import warnings
#import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import logging
import datetime
#from pynwb import NWBHDF5IO, NWBFile

FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(level='INFO', format=FORMAT, datefmt='%d-%b-%y %H:%M:%S')
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=ResourceWarning)

schema = dj.schema("mcoulter_realtime_error")

# parameters to calculate place field
@schema
class RealtimeErrorParameters(dj.Manual):
    definition = """
    realtime_error_param_name : varchar(500)
    ---
    realtime_error_parameters : mediumblob 
    """

# can include description above
#description : varchar(500)

    def insert_default(self):
        realtime_error_parameters = {}
        realtime_error_parameters['GPU_ID'] = 7
        realtime_error_parameters['speed_filter'] = 4
        realtime_error_parameters['sampling_frequency'] = 500
        realtime_error_parameters['xmin'] = 20
        realtime_error_parameters['xmax'] = 180
        realtime_error_parameters['ymin'] = 30
        realtime_error_parameters['ymax'] = 160
        self.insert1(['default', realtime_error_parameters], skip_duplicates=True)
    
# i dont think we need samplecount or statescriptfile in this list 
# select parameters and cluster 
@schema  
class RealtimeErrorSelection(dj.Manual):
    definition = """
    -> RealtimeFileImport
    -> RealtimeErrorParameters
    ---
    """

# items in the key
    #nwb_file_name
    #position_name
    #interval_list_name_pos ('pos 1 valid times')
    #interval_list_name_spikes ('02_r1')
    #position_interval_name

# this is the computed table - basically it has the results of the analysis
@schema 
class RealtimeError(dj.Computed):
    definition = """
    -> RealtimeErrorSelection
    ---
    reward_times : blob
    reward_times_2 : blob
    offset_diff : blob
    offset : double

    """

    def make(self, key):
        print(f'Computing task times for: {key}')
        
    # combine all steps for posterior fraction by position

    #ginny/tonks/molly/arthur: center well is at position 4
    # ron/george and pippin: center well is at position 3
    center_well_bin = 3

    # array to hold results: position x arm visit
    posterior_fraction_array = np.zeros((25,8,200))

    # this is the loop for the run sessions
    #for session in np.arange(0,186):
        
        # comment out if just using 1 hdf file
        hdf_file = pippin_file_list[session]

        store = pd.HDFStore(hdf_file, mode='r')

        decoder_data = pd.read_hdf(hdf_file,key='rec_4')
        occupancy_data = pd.read_hdf(hdf_file,key='rec_7')
        
        #encoder_data = pd.read_hdf(hdf_file,key='rec_3')
        #decoder_data = pd.read_hdf(hdf_file,key='rec_4')
        #likelihood_data = pd.read_hdf(hdf_file,key='rec_6')
        ##decoder_missed_spikes = store['rec_5']
        ##ripple_data = store['rec_1']
        ##stim_state = store['rec_10']
        #stim_lockout = pd.read_hdf(hdf_file,key='rec_11')
        #stim_message = pd.read_hdf(hdf_file,key='rec_12')
        #occupancy_data = pd.read_hdf(hdf_file,key='rec_7')

        print(hdf_file)
        
        # only do analysis if there is taskstate2
        if decoder_data[decoder_data['taskState']==2].shape[0]>0:
            
            task2_start_ts = decoder_data[decoder_data['taskState']==2][0:1]['bin_timestamp'].values[0]

            # this cell makes intervals for each position bin from occupancy_data
            occupancy_data_1 = occupancy_data.copy()
            position_intervals = occupancy_data_1.reset_index().groupby(
                        (occupancy_data_1['linear_pos'] != occupancy_data_1['linear_pos'].shift(
                        )).cumsum(), as_index = False) \
                        .agg({'linear_pos':[('position','first')],'index':[('first','first'),('last','last')]})
            position_intervals['start_ts'] = occupancy_data_1.iloc[
                position_intervals[('index','first')].values.tolist()]['bin_timestamp'].values
            position_intervals['end_ts'] = occupancy_data_1.iloc[
                position_intervals[('index','last')].values.tolist()]['bin_timestamp'].values

            # try to only include the outbound run - decoding is generally better


            # loop through visits to arm1 and find interval for whole run down arm
            # NOTE: does not include last run 
            intervals_24 = position_intervals[position_intervals[('linear_pos','position')]==24]
            arm1_runs = np.zeros((intervals_24.shape[0],2))
            arm1_in_runs = np.zeros((intervals_24.shape[0],2))
            arm1_out_runs = np.zeros((intervals_24.shape[0],2))

            for arm_end in np.arange(intervals_24.shape[0]-1):
                leave_arm_end_index = intervals_24.iloc[[arm_end]][('index','last')].values[0]
                if arm_end == intervals_24.shape[0]-1:
                    #arm1_runs[arm_end,1] = occupancy_data_1.iloc[[leave_arm_end_index]]['bin_timestamp'].values
                    #print('last one')
                    pass
                else:
                    for i in np.arange(occupancy_data.shape[0]):
                        #print(occupancy_data_1.iloc[[leave_arm_end_index+i]]['linear_pos'].values)
                        if (leave_arm_end_index+i)<occupancy_data_1.shape[0]:
                            if ((occupancy_data_1.iloc[[leave_arm_end_index+i]]['linear_pos'].values) == center_well_bin):
                                #print('arm end visit',arm_end,i,'center well',intervals_24.shape[0])

                                arm1_runs[arm_end,1] = occupancy_data_1.iloc[[leave_arm_end_index+i]]['bin_timestamp'].values
                                arm1_in_runs[arm_end,1] = occupancy_data_1.iloc[[leave_arm_end_index+i]]['bin_timestamp'].values
                                arm1_in_runs[arm_end,0] = np.mean([intervals_24['end_ts'].values[arm_end],
                                                           intervals_24['start_ts'].values[arm_end]])
                                break
                        else:
                            print('not center at end')
                            break

            for arm_end in np.arange(intervals_24.shape[0]):
                leave_arm_end_index = intervals_24.iloc[[arm_end]][('index','first')].values[0]
                if arm_end == intervals_24.shape[0]-1:
                    pass
                else:
                    for i in np.arange(occupancy_data.shape[0]):
                        #print(occupancy_data_1.iloc[[leave_arm_end_index+i]]['linear_pos'].values)
                        if (leave_arm_end_index+i)<occupancy_data_1.shape[0]:
                            if ((occupancy_data_1.iloc[[leave_arm_end_index-i]]['linear_pos'].values) == center_well_bin):
                                #print('arm end visit',arm_end,i,'center well')

                                arm1_runs[arm_end,0] = occupancy_data_1.iloc[[leave_arm_end_index-i]]['bin_timestamp'].values
                                arm1_out_runs[arm_end,0] = occupancy_data_1.iloc[[leave_arm_end_index-i]]['bin_timestamp'].values
                                arm1_out_runs[arm_end,1] = np.mean([intervals_24['end_ts'].values[arm_end],
                                                           intervals_24['start_ts'].values[arm_end]])                    
                                break 
                        else:
                            print('not center at end')
                            break

            arm1_runs
            arm1_runs_pandas = pd.DataFrame(arm1_runs, columns = ['start','end'])
            arm1_in_runs_pandas = pd.DataFrame(arm1_in_runs, columns = ['start','end'])
            arm1_out_runs_pandas = pd.DataFrame(arm1_out_runs, columns = ['start','end'])

            # loop through visits to arm2 and find interval for whole run down arm
            # NOTE: does not include last run
            intervals_40 = position_intervals[position_intervals[('linear_pos','position')]==40]
            arm2_runs = np.zeros((intervals_40.shape[0],2))
            arm2_in_runs = np.zeros((intervals_40.shape[0],2))
            arm2_out_runs = np.zeros((intervals_40.shape[0],2))

            for arm_end in np.arange(intervals_40.shape[0]):
                leave_arm_end_index = intervals_40.iloc[[arm_end]][('index','last')].values[0]
                if arm_end == intervals_40.shape[0]-1:
                    pass
                else:
                    for i in np.arange(occupancy_data.shape[0]):
                        #print(occupancy_data_1.iloc[[leave_arm_end_index+i]]['linear_pos'].values)
                        if (leave_arm_end_index+i)<occupancy_data_1.shape[0]:
                            if ((occupancy_data_1.iloc[[leave_arm_end_index+i]]['linear_pos'].values) == center_well_bin):
                                #print('arm end visit',arm_end,i,'center well')

                                arm2_runs[arm_end,1] = occupancy_data_1.iloc[[leave_arm_end_index+i]]['bin_timestamp'].values
                                arm2_in_runs[arm_end,1] = occupancy_data_1.iloc[[leave_arm_end_index+i]]['bin_timestamp'].values
                                arm2_in_runs[arm_end,0] = np.mean([intervals_40['end_ts'].values[arm_end],
                                                           intervals_40['start_ts'].values[arm_end]])
                                break
                        else:
                            print('not center at end')
                            break

            for arm_end in np.arange(intervals_40.shape[0]):
                leave_arm_end_index = intervals_40.iloc[[arm_end]][('index','first')].values[0]
                if arm_end == intervals_40.shape[0]-1:
                    pass
                else:
                    for i in np.arange(occupancy_data.shape[0]):
                        #print(occupancy_data_1.iloc[[leave_arm_end_index+i]]['linear_pos'].values)
                        if (leave_arm_end_index+i)<occupancy_data_1.shape[0]:
                            if ((occupancy_data_1.iloc[[leave_arm_end_index-i]]['linear_pos'].values) == center_well_bin):
                                #print('arm end visit',arm_end,i,'center well')

                                arm2_runs[arm_end,0] = occupancy_data_1.iloc[[leave_arm_end_index-i]]['bin_timestamp'].values
                                arm2_out_runs[arm_end,0] = occupancy_data_1.iloc[[leave_arm_end_index-i]]['bin_timestamp'].values
                                arm2_out_runs[arm_end,1] = np.mean([intervals_40['end_ts'].values[arm_end],
                                                           intervals_40['start_ts'].values[arm_end]]) 
                                break 
                        else:
                            print('not center at end')
                            break

            arm2_runs
            arm2_runs_pandas = pd.DataFrame(arm2_runs, columns = ['start','end'])
            arm2_in_runs_pandas = pd.DataFrame(arm2_in_runs, columns = ['start','end'])
            arm2_out_runs_pandas = pd.DataFrame(arm2_out_runs, columns = ['start','end'])

            # full arm runs
            #arm1_last_three = arm1_runs_pandas[(arm1_runs_pandas['start']<task2_start_ts)&(arm1_runs_pandas['start']>1)][-3:]
            #arm2_last_three = arm2_runs_pandas[(arm2_runs_pandas['start']<task2_start_ts)&(arm2_runs_pandas['start']>1)][-3:]
            # outbound arm runs
            arm1_last_three = arm1_out_runs_pandas[(arm1_out_runs_pandas['start']<task2_start_ts)&(arm1_out_runs_pandas['start']>1)][-3:]
            arm2_last_three = arm2_out_runs_pandas[(arm2_out_runs_pandas['start']<task2_start_ts)&(arm2_out_runs_pandas['start']>1)][-3:]
            # inbound arm runs
            #arm1_last_three = arm1_in_runs_pandas[(arm1_in_runs_pandas['start']<task2_start_ts)&(arm1_in_runs_pandas['start']>1)][-3:]
            #arm2_last_three = arm2_in_runs_pandas[(arm2_in_runs_pandas['start']<task2_start_ts)&(arm2_in_runs_pandas['start']>1)][-3:]


            # test with first 3 visits
            #arm1_last_three = arm1_runs_pandas[(arm1_runs_pandas['start']<task2_start_ts)&(arm1_runs_pandas['start']>1)][0:3]
            #arm2_last_three = arm2_runs_pandas[(arm2_runs_pandas['start']<task2_start_ts)&(arm2_runs_pandas['start']>1)][0:3]

            import statistics

            #check that arms exist
            if arm1_last_three.shape[0]>0 and arm2_last_three.shape[0]>0:
                
                # first get average timebin for position 23 and 39
                bin_23_1 = (decoder_data[(decoder_data['bin_timestamp']>arm1_last_three['start'].values[0])&
                                               (decoder_data['bin_timestamp']<arm1_last_three['end'].values[0])&
                                              (decoder_data['velocity']>10)&(decoder_data['real_pos']==23)].shape[0])
                bin_23_2 = (decoder_data[(decoder_data['bin_timestamp']>arm1_last_three['start'].values[1])&
                                               (decoder_data['bin_timestamp']<arm1_last_three['end'].values[1])&
                                              (decoder_data['velocity']>10)&(decoder_data['real_pos']==23)].shape[0])
                bin_23_3 = (decoder_data[(decoder_data['bin_timestamp']>arm1_last_three['start'].values[2])&
                                               (decoder_data['bin_timestamp']<arm1_last_three['end'].values[2])&
                                              (decoder_data['velocity']>10)&(decoder_data['real_pos']==23)].shape[0])

                bin_23_avg = int(np.average([bin_23_1,bin_23_2,bin_23_3])/2)
                #statistics.mean([])
                print('position 23 average time bins',bin_23_avg)

                bin_39_1 = (decoder_data[(decoder_data['bin_timestamp']>arm2_last_three['start'].values[0])&
                                               (decoder_data['bin_timestamp']<arm2_last_three['end'].values[0])&
                                              (decoder_data['velocity']>10)&(decoder_data['real_pos']==39)].shape[0])
                bin_39_2 = (decoder_data[(decoder_data['bin_timestamp']>arm2_last_three['start'].values[1])&
                                               (decoder_data['bin_timestamp']<arm2_last_three['end'].values[1])&
                                              (decoder_data['velocity']>10)&(decoder_data['real_pos']==39)].shape[0])
                bin_39_3 = (decoder_data[(decoder_data['bin_timestamp']>arm2_last_three['start'].values[2])&
                                               (decoder_data['bin_timestamp']<arm2_last_three['end'].values[2])&
                                              (decoder_data['velocity']>10)&(decoder_data['real_pos']==39)].shape[0])

                bin_39_avg = int(np.average([bin_39_1,bin_39_2,bin_39_3])/2)
                print('position 39 average time bins',bin_39_avg)

                # arm 1          
                for j in np.arange(3):
                    arm_run_decoder = decoder_data[(decoder_data['bin_timestamp']>arm1_last_three['start'].values[j])&
                                                   (decoder_data['bin_timestamp']<arm1_last_three['end'].values[j])&
                                                  (decoder_data['velocity']>10)]
                    arm_run_decoder['post_max'] = arm_run_decoder.iloc[:, 27:68].idxmax(axis=1).str.slice(1,3,1).astype('int16')

                    # want to only use beginning and end times at the well, can use average of next to last position
                    # print(arm_run_decoder[arm_run_decoder['real_pos']==23].shape[0]) - divide this by 2
                    # ideally find this number for all 3 runs and average all
                    # then if position == 24, only do calcuation on a subset of it

                    # select columns by index: df.iloc[:, 0:3]
                    for i in np.arange(13,25):
                        posterior_fraction_array[i-13,0,session] = i
                        new_dataframe = arm_run_decoder[arm_run_decoder['real_pos']==i]
                        #print(new_dataframe.shape)
                        # this was to calculate fraction posterior +/- 2 bins of real position

                        # error distance (from post max to actual pos when spike > 0)
                        if i == 24:
                            new_dataframe_begin = new_dataframe[:bin_23_avg]
                            new_dataframe_end = new_dataframe[-bin_23_avg:]
                            #new_dataframe_2 = pd.concat(new_dataframe_begin,new_dataframe_end)
                            new_dataframe_2 = new_dataframe_begin.append(new_dataframe_end, ignore_index=True)
                            new_dataframe_3 = new_dataframe_2[new_dataframe_2['spike_count']>0]
                            # this is distance for any position bin
                            posterior_position_dist = np.average(new_dataframe_3['real_pos'] - new_dataframe_3['post_max'])*5
                            # this is to record where max is in other arm or box
                            local_bins = new_dataframe_3[(new_dataframe_3['post_max']>10)&
                                                         (new_dataframe_3['post_max']<26)].shape[0]
                            #only do this is shape > 0
                            if new_dataframe_3.shape[0] > 0:
                                posterior_fraction_array[i-13,j+4,session] = 1-(local_bins/new_dataframe_3.shape[0])
                                posterior_fraction_array[i-13,j+1,session] = new_dataframe_3.shape[0]
                                # posterior distance
                                #posterior_fraction_array[i-13,j+4,session] = posterior_position_dist

                            #check by plotting position
                            #plt.figure(figsize=(8,3))
                            #plt.scatter(arm_run_decoder['bin_timestamp'].values,arm_run_decoder['real_pos'].values,s=1,color='black')
                            #plt.scatter(new_dataframe_2['bin_timestamp'].values,new_dataframe_2['real_pos'].values,s=1,color='red')

                        else:
                            new_dataframe_3 = new_dataframe[new_dataframe['spike_count']>0]
                            # for +/- 2 position bins: 36,41
                            posterior_position_dist = np.average(new_dataframe_3['real_pos'] - new_dataframe_3['post_max'])*5
                            #print('position',i,'posterior fraction',posterior_fraction_position)
                            posterior_fraction_array[i-13,j+1,session] = new_dataframe_3.shape[0]
                            #posterior distance
                            #posterior_fraction_array[i-13,j+4,session] = posterior_position_dist
                            local_bins = new_dataframe_3[(new_dataframe_3['post_max']>10)&
                                                         (new_dataframe_3['post_max']<26)].shape[0]
                            #only do this is shape > 0
                            if new_dataframe_3.shape[0] > 0:                
                                posterior_fraction_array[i-13,j+4,session] = 1-(local_bins/new_dataframe_3.shape[0])                

                # arm 2
                for j in np.arange(3):
                    arm_run_decoder = decoder_data[(decoder_data['bin_timestamp']>arm2_last_three['start'].values[j])&
                                                   (decoder_data['bin_timestamp']<arm2_last_three['end'].values[j])&
                                                  (decoder_data['velocity']>10)]
                    arm_run_decoder['post_max'] = arm_run_decoder.iloc[:, 27:68].idxmax(axis=1).str.slice(1,3,1).astype('int16')

                    # select columns by index: df.iloc[:, 0:3]
                    for i in np.arange(29,41):
                        posterior_fraction_array[i-16,0,session] = i
                        new_dataframe = arm_run_decoder[arm_run_decoder['real_pos']==i]
                        #print(i,new_dataframe.shape)
                        posterior_fraction_array[i-16,j+1,session] = new_dataframe.shape[0]

                        # distance from position to posterior max
                        # need to do update below
                        if i == 39:
                            new_dataframe_3 = new_dataframe[new_dataframe['spike_count']>0]
                            # for +/- 2 position bins: 36,41
                            posterior_position_dist = np.average(new_dataframe_3['real_pos'] - new_dataframe_3['post_max'])*5
                            #print('position',i,'posterior fraction',posterior_fraction_position)
                            posterior_fraction_array[i-16,j+1,session] = new_dataframe_3.shape[0]
                            #posterior distance
                            #posterior_fraction_array[i-16,j+4,session] = posterior_position_dist
                            local_bins = new_dataframe_3[(new_dataframe_3['post_max']>26)&
                                                         (new_dataframe_3['post_max']<41)].shape[0]
                            #only do this is shape > 0
                            if new_dataframe_3.shape[0] > 0:
                                posterior_fraction_array[i-16,j+4,session] = 1-(local_bins/new_dataframe_3.shape[0])

                        elif i == 40:
                            new_dataframe_begin = new_dataframe[:bin_39_avg]
                            new_dataframe_end = new_dataframe[-bin_39_avg:]
                            #new_dataframe_2 = pd.concat(new_dataframe_begin,new_dataframe_end)
                            new_dataframe_2 = new_dataframe_begin.append(new_dataframe_end, ignore_index=True)
                            new_dataframe_3 = new_dataframe_2[new_dataframe_2['spike_count']>0]
                            posterior_position_dist = np.average(new_dataframe_3['real_pos'] - new_dataframe_3['post_max'])*5
                            posterior_fraction_array[i-16,j+1,session] = new_dataframe_3.shape[0]
                            #postior distance
                            #posterior_fraction_array[i-16,j+4,session] = posterior_position_dist
                            local_bins = new_dataframe_3[(new_dataframe_3['post_max']>26)&
                                                         (new_dataframe_3['post_max']<41)].shape[0]
                            #only do this is shape > 0
                            if new_dataframe_3.shape[0] > 0:
                                posterior_fraction_array[i-16,j+4,session] = 1-(local_bins/new_dataframe_3.shape[0])

                            #check by plotting position
                            #plt.figure(figsize=(8,3))
                            #plt.scatter(arm_run_decoder['bin_timestamp'].values,arm_run_decoder['real_pos'].values,s=1,color='black')
                            #plt.scatter(new_dataframe_2['bin_timestamp'].values,new_dataframe_2['real_pos'].values,s=1,color='red')

                        else:                    
                            new_dataframe_3 = new_dataframe[new_dataframe['spike_count']>0]
                            # for +/- 2 position bins: 36,41
                            posterior_position_dist = np.average(new_dataframe_3['real_pos'] - new_dataframe_3['post_max'])*5
                            #print('position',i,'posterior fraction',posterior_fraction_position)
                            posterior_fraction_array[i-16,j+1,session] = new_dataframe_3.shape[0]
                            #posterior distance
                            #posterior_fraction_array[i-16,j+4,session] = posterior_position_dist
                            local_bins = new_dataframe_3[(new_dataframe_3['post_max']>26)&
                                                         (new_dataframe_3['post_max']<41)].shape[0]
                            #only do this is shape > 0
                            if new_dataframe_3.shape[0] > 0:                
                                posterior_fraction_array[i-16,j+4,session] = 1-(local_bins/new_dataframe_3.shape[0])

                        #print('position',i,'posterior fraction',posterior_fraction_position)
                        #posterior_fraction_array[i-16,j+4,session] = posterior_fraction_position


        key['reward_times'] = unix_sound_cue
        key['reward_times_2'] = unix_sound_cue_2
        key['offset_diff'] = np.diff(time_offset)
        key['offset'] = average_time_offset

        self.insert1(key)
