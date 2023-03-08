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

schema = dj.schema("mcoulter_realtime_posterior_sum")

# parameters to calculate place field
@schema
class RealtimePosteriorSumParameters(dj.Manual):
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
class RealtimePosteriorSumSelection(dj.Manual):
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
class RealtimePosteriorSum(dj.Computed):
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

    # cell to summarize results from sessions listed in filelist
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx],idx

    # dictionaries for head direction
    arm1_angles_dict = {}
    arm2_angles_dict = {}
    arm1_x1_dict = {}
    arm1_y1_dict = {}
    arm2_x1_dict = {}
    arm2_y1_dict = {}

    plot_on = False
    plot_2 = False

    all_task2_head_dir = np.zeros((200,))
    arm1_head_dir_list = np.zeros((200,))
    arm2_head_dir_list = np.zeros((200,))
    arm1_end = np.zeros((200,))
    arm2_end = np.zeros((200,))
    arm1_total = np.zeros((200,))
    arm2_total = np.zeros((200,))
    arm1_events_2tet = np.zeros((200,))
    arm2_events_2tet = np.zeros((200,))
    arm1_event_rate_2tet = np.zeros((200,))
    arm2_event_rate_2tet = np.zeros((200,))
    arm1_events_1tet = np.zeros((200,))
    arm2_events_1tet = np.zeros((200,))
    arm1_event_rate_1tet = np.zeros((200,))
    arm2_event_rate_1tet = np.zeros((200,))
    reward_counts = np.zeros((200,))
    reward_rate = np.zeros((200,))
    arm1_no_rew_event_rate_2tet = np.zeros((200,))
    arm2_no_rew_event_rate_2tet = np.zeros((200,))
    arm1_no_rew_events_2tet = np.zeros((200,))
    arm2_no_rew_events_2tet = np.zeros((200,))
    non_engaged_time = np.zeros((200,))
    high_spike_fraction = np.zeros((200,))
    task2_center_velocity = np.zeros((200,))

    well_distance_max = 17

    # ron & george head direction: file_list[0:12] and file_list[75:87]
    # ginny & tonks head direction: file_list[0:12] and file_list[66:78]
    for i_session in np.arange(0,40):
        
        hdf_file = pippin_file_list[i_session]

        #store = pd.HDFStore(hdf_file, mode='r')

        decoder_data = pd.read_hdf(hdf_file,key='rec_4')
        occupancy_data = pd.read_hdf(hdf_file,key='rec_7')
        encoder_data = pd.read_hdf(hdf_file,key='rec_3')
        #likelihood_data = pd.read_hdf(hdf_file,key='rec_6')
        ##decoder_missed_spikes = store['rec_5']
        ##ripple_data = store['rec_1']
        ##stim_state = store['rec_10']
        stim_lockout = pd.read_hdf(hdf_file,key='rec_11')
        stim_message = pd.read_hdf(hdf_file,key='rec_12')
        
        # ron/george
        #if i_session in [0,1,2,3,4,5,6,7,8,9,10,11,75,76,77,78,79,80,81,82,83,84,85,86]:
        #    head_dir_data = store['rec_13']
        # ginny/tonks
        #if i_session in [0,1,2,3,4,5,6,7,8,9,10,11,66,67,68,69,70,71,72,73,74,75,76,77]:
        #    head_dir_data = store['rec_13']
        
        
        print(hdf_file)
        print(encoder_data.shape)
        #print(encoder_data[encoder_data['encode_spike']==1].shape)

        # reward count
        reward_counts[i_session] = stim_message[stim_message['shortcut_message_sent']==1].shape[0]
        print('rewards',stim_message[stim_message['shortcut_message_sent']==1].shape[0])
        #print('instructive rewards',np.count_nonzero(reward_list[i_session,:]))
        # fred, pippin, percy
        center_well_pos = [634,648]
        # geroge, ron, ginny, tonks
        #center_well_pos = [449,330]
        
        decoder_data_center_well = decoder_data.copy()
        decoder_data_center_well['center_well_dist'] = np.sqrt(np.square(decoder_data_center_well['smooth_x']-
                                                                         center_well_pos[0]) + 
                                                               np.square(decoder_data_center_well['smooth_y']
                                                                         -center_well_pos[1]))*0.222
        decoder_data_center_well['arm1_end'] = (decoder_data_center_well['x20']+decoder_data_center_well['x21']+
                                               decoder_data_center_well['x22']+decoder_data_center_well['x23']+
                                               decoder_data_center_well['x24'])
        decoder_data_center_well['arm2_end'] = (decoder_data_center_well['x36']+decoder_data_center_well['x37']+
                                               decoder_data_center_well['x38']+decoder_data_center_well['x39']+
                                               decoder_data_center_well['x40'])
        decoder_data_center_well['arm1_all'] = (decoder_data_center_well['x13']+decoder_data_center_well['x14']+
                                                decoder_data_center_well['x15']+decoder_data_center_well['x16']+
                                                decoder_data_center_well['x17']+decoder_data_center_well['x18']+
                                                decoder_data_center_well['x19']+
                                                decoder_data_center_well['x20']+decoder_data_center_well['x21']+
                                               decoder_data_center_well['x22']+decoder_data_center_well['x23']+
                                               decoder_data_center_well['x24'])
        decoder_data_center_well['arm2_all'] = (decoder_data_center_well['x29']+decoder_data_center_well['x30']+
                                                decoder_data_center_well['x31']+decoder_data_center_well['x32']+
                                                decoder_data_center_well['x33']+decoder_data_center_well['x34']+
                                                decoder_data_center_well['x35']+
                                                decoder_data_center_well['x36']+decoder_data_center_well['x37']+
                                               decoder_data_center_well['x38']+decoder_data_center_well['x39']+
                                               decoder_data_center_well['x40'])    

        task2_decode_center = decoder_data_center_well[(decoder_data_center_well['center_well_dist']<well_distance_max)&
                                (decoder_data_center_well['taskState']==2)]
        #task2_decode_center = decoder_data_center_well[(decoder_data_center_well['real_pos']<8)&
        #                        (decoder_data_center_well['taskState']==2)]
        task2_head_dir = pd.merge_asof(task2_decode_center,occupancy_data,on='bin_timestamp',direction='nearest')
        #print('timebins task2 near center',task2_head_dir.shape)

        #print('total arm1 end',task2_decode_center['arm1_end'].sum()/task2_decode_center.shape[0])
        #print('total arm2 end',task2_decode_center['arm2_end'].sum()/task2_decode_center.shape[0])

        #print(hdf_file)
        
        # number of bins with >0.4 of posterior in arm end
        if task2_decode_center.shape[0] > 0:
            #print('>0.4 arm1 end',task2_decode_center[task2_decode_center['arm1_end']>0.4].shape[0]
            #                  /task2_decode_center.shape[0])
            #print('>0.4 arm2 end',task2_decode_center[task2_decode_center['arm2_end']>0.4].shape[0]
            #                  /task2_decode_center.shape[0])
            #arm1_end[i_session] = (task2_decode_center[task2_decode_center['arm1_end']>0.4].shape[0]
            #                  /task2_decode_center.shape[0])
            #arm2_end[i_session] = (task2_decode_center[task2_decode_center['arm2_end']>0.4].shape[0]
            #                  /task2_decode_center.shape[0])
            #arm1_total[i_session] = task2_decode_center['arm1_all'].sum()/task2_decode_center.shape[0]
            #arm2_total[i_session] = task2_decode_center['arm2_all'].sum()/task2_decode_center.shape[0]
            
            # try to remove time bins with <3 spikes (for molly <10 spikes)
            # this will be interesting to calculate either way: divide by all bins or just bins with spikes
            #task2_decode_center_spikes = task2_decode_center[(task2_decode_center['spike_count']>2)]
            #print('spikes > 2',task2_decode_center_spikes.shape[0], 'all bins',task2_decode_center.shape[0])
            #print('>0.4 arm1 end',task2_decode_center_spikes[task2_decode_center_spikes['arm1_end']>0.4].shape[0]
            #                  /task2_decode_center.shape[0])
            #print('>0.4 arm2 end',task2_decode_center_spikes[task2_decode_center_spikes['arm2_end']>0.4].shape[0]
            #                  /task2_decode_center.shape[0])
            #arm1_end[i_session] = (task2_decode_center_spikes[task2_decode_center_spikes['arm1_end']>0.4].shape[0]
            #                  /task2_decode_center.shape[0])
            #arm2_end[i_session] = (task2_decode_center_spikes[task2_decode_center_spikes['arm2_end']>0.4].shape[0]
            #                  /task2_decode_center.shape[0])
            arm1_total[i_session] = task2_decode_center_spikes['arm1_all'].sum()/task2_decode_center.shape[0]
            arm2_total[i_session] = task2_decode_center_spikes['arm2_all'].sum()/task2_decode_center.shape[0]    
            
            # after spike filtering try summing total decode 
            task2_decode_center_spikes = task2_decode_center[(task2_decode_center['spike_count']>2)]
            print('spikes > 2',task2_decode_center_spikes.shape[0], 'all bins',task2_decode_center.shape[0])
            print('total decode arm1 end',task2_decode_center_spikes.iloc[:,47:52].sum(axis=1).sum()/task2_decode_center_spikes.shape[0])
            print('total decode arm2 end',task2_decode_center_spikes.iloc[:,63:68].sum(axis=1).sum()/task2_decode_center_spikes.shape[0])
            arm1_end[i_session] = task2_decode_center_spikes.iloc[:,47:52].sum(axis=1).sum()/task2_decode_center_spikes.shape[0]
            arm2_end[i_session] = task2_decode_center_spikes.iloc[:,63:68].sum(axis=1).sum()/task2_decode_center_spikes.shape[0]
            # arm2 end
            #task2_decode_center_spikes.iloc[:,63:68].sum(axis=1).sum()/task2_decode_center_spikes.shape[0]
            # arm1 end
            #task2_decode_center_spikes.iloc[:,47:52].sum(axis=1).sum()/task2_decode_center_spikes.shape[0]
            
            # save fraction of task2 time with good spiking
            high_spike_fraction[i_session] = task2_decode_center_spikes.shape[0]/task2_decode_center.shape[0]
            
            # save average velocity during task2 time
            task2_center_velocity[i_session] = np.mean(task2_decode_center['velocity'])
            
        # event counts for all or >1 tets
        #print('arm1 all:',stim_message[(stim_message['taskState']==2) & (stim_message['real_pos']<8) & 
        #                       (stim_message['real_pos']>=0) &
        #         (stim_message['posterior_max_arm']==1)& (stim_message['unique_tets']>0)].shape[0])
        #print('arm2 all:',stim_message[(stim_message['taskState']==2) & (stim_message['real_pos']<8) &
        #                       (stim_message['real_pos']>=0) &
        #         (stim_message['posterior_max_arm']==2)& (stim_message['unique_tets']>0)].shape[0])
        
        print('arm1 >1 tet:',stim_message[(stim_message['taskState']==2) & (stim_message['center_well_dist']<well_distance_max) &
             (stim_message['posterior_max_arm']==1)& (stim_message['unique_tets']>1)].shape[0])
        print('arm2 >1 tet:',stim_message[(stim_message['taskState']==2) & (stim_message['center_well_dist']<well_distance_max) &
             (stim_message['posterior_max_arm']==2)& (stim_message['unique_tets']>1)].shape[0])
        arm1_events_2tet[i_session] = stim_message[(stim_message['taskState']==2) & (stim_message['center_well_dist']<well_distance_max) &
             (stim_message['posterior_max_arm']==1)& (stim_message['unique_tets']>1)].shape[0]
        arm2_events_2tet[i_session] = stim_message[(stim_message['taskState']==2) & (stim_message['center_well_dist']<well_distance_max) &
             (stim_message['posterior_max_arm']==2)& (stim_message['unique_tets']>1)].shape[0]    

        # also make event counts for 1 tet 
        arm1_events_1tet[i_session] = stim_message[(stim_message['taskState']==2) & (stim_message['center_well_dist']<well_distance_max) &
                 (stim_message['posterior_max_arm']==1)& (stim_message['unique_tets']>0)].shape[0]
        arm2_events_1tet[i_session] = stim_message[(stim_message['taskState']==2) & (stim_message['center_well_dist']<well_distance_max) &
                 (stim_message['posterior_max_arm']==2)& (stim_message['unique_tets']>0)].shape[0]    

        # event counts with post-reward filter (10 sec = 300,000 timestamps)
        post_reward_events = []
        # make reward time list from head direction
        reward_counter = 0
        task2_decoder = decoder_data[decoder_data['taskState']==2]
        # ron/george
        #if i_session in [0,1,2,3,4,5,6,7,8,9,10,11,75,76,77,78,79,80,81,82,83,84,85,86]:
        # ginny/tonks
        #if i_session in [0,1,2,3,4,5,6,7,8,9,10,11,66,67,68,69,70,71,72,73,74,75,76,77]:
        # timer sessions - head direction
        #if i_session == 300:
        #    for time in head_dir_data['timestamp']:
        #        if ((time<task2_decoder['bin_timestamp'].values[-1]) and (time>task2_decoder['bin_timestamp'].values[0])):
        #            reward_counter += 1
        #        post_reward_events.append(stim_message[(stim_message['bin_timestamp']>time)&
        #                           (stim_message['bin_timestamp']<(time+(30000*5)))].index.tolist())
        #else:
        #    for time in stim_message[stim_message['shortcut_message_sent']==1]['bin_timestamp']:
        #        if ((time<task2_decoder['bin_timestamp'].values[-1]) and (time>task2_decoder['bin_timestamp'].values[0])):
        #            reward_counter += 1
        #        post_reward_events.append(stim_message[(stim_message['bin_timestamp']>time)&
        #                           (stim_message['bin_timestamp']<(time+(30000*5)))].index.tolist())
                
        # instructive task for ron
        for time in stim_message[stim_message['shortcut_message_sent']==1]['bin_timestamp']:
            post_reward_events.append(stim_message[(stim_message['bin_timestamp']>time)&
                               (stim_message['bin_timestamp']<(time+300000))].index.tolist())
        # use statescript for reward times
        #reward_list[0,:]*30
        #for time in reward_list[i_session,:]*30:
        #    post_reward_events.append(stim_message[(stim_message['bin_timestamp']>time)&
        #                       (stim_message['bin_timestamp']<(time+300000))].index.tolist())            
        #post_reward_events
        post_reward_events_flat = [item for sublist in post_reward_events for item in sublist]
        #post_reward_events_flat
        stim_message_no_reward = stim_message.loc[stim_message.index.difference(post_reward_events_flat)]
        #stim_message_no_reward[(stim_message_no_reward['taskState']==2)&(stim_message_no_reward['center_well_dist']<17)
        #            &(stim_message_no_reward['unique_tets']>1)]

        # no-reward 2 tet events - turn off for percy
        print('arm1 no reawrd', stim_message_no_reward[(stim_message_no_reward['taskState']==2) & 
                                                (stim_message_no_reward['center_well_dist']<well_distance_max) &
                 (stim_message_no_reward['posterior_max_arm']==1)& (stim_message_no_reward['unique_tets']>1)].shape[0])
        print('arm2 no reward', stim_message_no_reward[(stim_message_no_reward['taskState']==2) & 
                                                (stim_message_no_reward['center_well_dist']<well_distance_max) &
                 (stim_message_no_reward['posterior_max_arm']==2)& (stim_message_no_reward['unique_tets']>1)].shape[0])
        arm1_no_rew_events_2tet[i_session] = stim_message_no_reward[(stim_message_no_reward['taskState']==2) & 
                                                          (stim_message_no_reward['center_well_dist']<well_distance_max) &
                 (stim_message_no_reward['posterior_max_arm']==1)& (stim_message_no_reward['unique_tets']>1)].shape[0]
        arm2_no_rew_events_2tet[i_session] = stim_message_no_reward[(stim_message_no_reward['taskState']==2) & 
                                                          (stim_message_no_reward['center_well_dist']<well_distance_max) &
                 (stim_message_no_reward['posterior_max_arm']==2)& (stim_message_no_reward['unique_tets']>1)].shape[0]     
        
        # task2 time in minutes
        # now for no reward we subtract the reward comsumption time from the task2 time
        task2_decoder_all = decoder_data[decoder_data['taskState']==2]
        # try only using time at center well
        task2_decoder = decoder_data_center_well[(decoder_data_center_well['center_well_dist']<well_distance_max)&
                                (decoder_data_center_well['taskState']==2)] 
        
        if task2_decoder.shape[0]>0:
            #task2_time = (task2_decoder['bin_timestamp'].values[-1] - task2_decoder['bin_timestamp'].values[0])/30000/60
            task2_time = task2_decoder.shape[0]/167/60
            task2_time_all = (task2_decoder_all['bin_timestamp'].values[-1] - task2_decoder_all['bin_timestamp'].values[0])/30000/60
            non_engaged_time[i_session] = (task2_time_all-task2_time)/task2_time_all
            print('fraction not at well',non_engaged_time[i_session])
            print('task2 start',task2_decoder['bin_timestamp'].values[0])
            print('task2 time all',task2_time_all)
            print('task2 time',task2_time, 'no reward time',(task2_time-(5/60*reward_counter)))
            arm1_event_rate_2tet[i_session] = arm1_events_2tet[i_session]/task2_time
            arm2_event_rate_2tet[i_session] = arm2_events_2tet[i_session]/task2_time
            arm1_event_rate_1tet[i_session] = arm1_events_1tet[i_session]/task2_time
            arm2_event_rate_1tet[i_session] = arm2_events_1tet[i_session]/task2_time
            print('reward count',reward_counter)
            arm1_no_rew_event_rate_2tet[i_session] = arm1_no_rew_events_2tet[i_session]/(task2_time-(5/60*reward_counter))
            arm2_no_rew_event_rate_2tet[i_session] = arm2_no_rew_events_2tet[i_session]/(task2_time-(5/60*reward_counter))        
            
            # for normal
            reward_rate[i_session] = stim_message[stim_message['shortcut_message_sent']==1].shape[0]/task2_time_all
            # for ron instructive - include all task2 time
            #reward_rate[i_session] = np.count_nonzero(reward_list[i_session,:])/task2_time_all
        
        #print('all message',stim_message[(stim_message['shortcut_message_sent']==1)].shape[0])
        #print('message < 15',stim_message[(stim_message['shortcut_message_sent']==1) &(stim_message['center_well_dist']<15) ].shape[0])
        
        # caclulate head direction
        monty_head_dir = occupancy_data[["bin_timestamp", "raw_x","raw_y","raw_x2","raw_y2"]].to_numpy() 

        # replaced pos < 8 and pos > 0 with center_well_dist < 20
        arm1_replay = stim_message[(stim_message['taskState']==2) & (stim_message['center_well_dist']<well_distance_max) & 
                                   (stim_message['unique_tets']>1) &(stim_message['posterior_max_arm']==1)]
        # test head dir while in arm 1
        #arm1_replay = occupancy_data[occupancy_data['linear_pos']==20]

        arm2_replay = stim_message[(stim_message['taskState']==2) & (stim_message['center_well_dist']<well_distance_max) & 
                                   (stim_message['unique_tets']>1) &(stim_message['posterior_max_arm']==2)]
        
        #merge stim_message with occupancy
        arm1_replay_occup = pd.merge_asof(arm1_replay,occupancy_data,on='bin_timestamp',direction='nearest')
        arm2_replay_occup = pd.merge_asof(arm2_replay,occupancy_data,on='bin_timestamp',direction='nearest')
        
        # add angle column
        arm1_replay_occup['angle_2'] = np.arctan2((arm1_replay_occup['raw_y2']-arm1_replay_occup['raw_y']),
                    (arm1_replay_occup['raw_x2']-arm1_replay_occup['raw_x']))*(180/np.pi)
        arm2_replay_occup['angle_2'] = np.arctan2((arm2_replay_occup['raw_y2']-arm2_replay_occup['raw_y']),
                    (arm2_replay_occup['raw_x2']-arm2_replay_occup['raw_x']))*(180/np.pi)
        arm1_head_dir_filter = arm1_replay_occup[(arm1_replay_occup['angle_2']>80)&(arm1_replay_occup['angle_2']<100)]
        arm2_head_dir_filter = arm2_replay_occup[(arm2_replay_occup['angle_2']>80)&(arm2_replay_occup['angle_2']<100)]
        arm1_head_dir_list[i_session] = arm1_head_dir_filter.shape[0]
        arm2_head_dir_list[i_session] = arm2_head_dir_filter.shape[0]
        #print('arm1. all',arm1_replay.shape[0],'head dir filter',arm1_head_dir_filter.shape[0])
        #print('arm2. all',arm2_replay.shape[0],'head dir filter',arm2_head_dir_filter.shape[0])
        
        # test head dir while in arm 1
        #arm2_replay = occupancy_data[occupancy_data['linear_pos']==35]

        #print('arm1',arm1_replay.shape)
        #print('arm2',arm2_replay.shape)

        # convert decoding timestamp to filterframework time
        arm1_times = (arm1_replay['bin_timestamp'].values)
        arm2_times = (arm2_replay['bin_timestamp'].values)

        arm1_replay_led = np.zeros((len(arm1_times),5))
        #print('arm1',arm1_replay_led.shape)
        for i in np.arange(len(arm1_times)):
            #print(find_nearest(monty_head_dir[:,0],arm1_times[i]))
            #print(monty_head_dir[idx,:])
            idx = find_nearest(monty_head_dir[:,0],arm1_times[i])[1]
            #print(idx)
            #print(monty_head_dir[idx,:])
            arm1_replay_led[i,:] = monty_head_dir[idx,:]

        arm2_replay_led = np.zeros((len(arm2_times),5))
        #print('arm2',arm2_replay_led.shape)
        for i in np.arange(len(arm2_times)):
            idx = find_nearest(monty_head_dir[:,0],arm2_times[i])[1]
            arm2_replay_led[i,:] = monty_head_dir[idx,:]

        # note: for arctan2 function, y-coord is first, x-coord is second
        task2_head_dir['angle_2'] = np.arctan2((task2_head_dir['raw_y2']-task2_head_dir['raw_y_y']),
                    (task2_head_dir['raw_x2']-task2_head_dir['raw_x_y']))
        
        all_task2_head_dir[i_session] = task2_head_dir['angle_2'].mean()

        unit_vectors_1 = np.zeros((len(arm1_replay_led),2))
        for i in np.arange(len(arm1_replay_led)):
            led_vector = np.asarray(((arm1_replay_led[i][3]-arm1_replay_led[i][1]),
                        (arm1_replay_led[i][4]-arm1_replay_led[i][2])))
            unit_vectors_1[i,:] = led_vector/np.linalg.norm(led_vector)
            #print(led_vector)
            #print(np.linalg.norm(led_vector))
            #print(led_vector/np.linalg.norm(led_vector))
        #print('arm1',unit_vectors_1.shape)
        unit_angles_1 = np.arctan2(*unit_vectors_1.T[::-1])

        unit_vectors_2 = np.zeros((len(arm2_replay_led),2))
        for i in np.arange(len(arm2_replay_led)):
            led_vector = np.asarray(((arm2_replay_led[i][3]-arm2_replay_led[i][1]),
                        (arm2_replay_led[i][4]-arm2_replay_led[i][2])))
            unit_vectors_2[i,:] = led_vector/np.linalg.norm(led_vector)
        #print('arm2',unit_vectors_2.shape)
        unit_angles_2 = np.arctan2(*unit_vectors_2.T[::-1])
        

        # add arm1 and arm2 angles to respective dictionaries
        arm1_angles_dict[hdf_file] = unit_angles_1
        arm2_angles_dict[hdf_file] = unit_angles_2

        arm1_x1_dict[hdf_file] = arm1_replay_led[:,1]
        arm1_y1_dict[hdf_file] = arm1_replay_led[:,2]
        arm2_x1_dict[hdf_file] = arm2_replay_led[:,1]
        arm2_y1_dict[hdf_file] = arm2_replay_led[:,2]
        
        # fred, pippin, percy
        #center_well_pos = [634,648]
        # geroge ron
        
        decoder_data_center_well = decoder_data.copy()
        decoder_data_center_well['center_well_dist'] = np.sqrt(np.square(decoder_data_center_well['smooth_x']-
                                                                         center_well_pos[0]) + 
                                                               np.square(decoder_data_center_well['smooth_y']
                                                                         -center_well_pos[1]))*0.222
        decoder_data_center_well['arm1_end'] = (decoder_data_center_well['x20']+decoder_data_center_well['x21']+
                                               decoder_data_center_well['x22']+decoder_data_center_well['x23']+
                                               decoder_data_center_well['x24'])
        decoder_data_center_well['arm2_end'] = (decoder_data_center_well['x36']+decoder_data_center_well['x37']+
                                               decoder_data_center_well['x38']+decoder_data_center_well['x39']+
                                               decoder_data_center_well['x40'])
        decoder_data_center_well['arm1_all'] = (
                                                decoder_data_center_well['x13']+decoder_data_center_well['x14']+
                                               decoder_data_center_well['x15']+decoder_data_center_well['x16']+
                                                decoder_data_center_well['x17']+decoder_data_center_well['x18']+
                                               decoder_data_center_well['x19']+
                                                decoder_data_center_well['x20']+decoder_data_center_well['x21']+
                                               decoder_data_center_well['x22']+decoder_data_center_well['x23']+
                                               decoder_data_center_well['x24'])
        decoder_data_center_well['arm2_all'] = (
                                                decoder_data_center_well['x29']+decoder_data_center_well['x30']+
                                               decoder_data_center_well['x31']+decoder_data_center_well['x32']+
                                                decoder_data_center_well['x33']+decoder_data_center_well['x34']+
                                               decoder_data_center_well['x35']+
                                                decoder_data_center_well['x36']+decoder_data_center_well['x37']+
                                               decoder_data_center_well['x38']+decoder_data_center_well['x39']+
                                               decoder_data_center_well['x40'])
        decoder_data_center_well['box_post'] = (
                                                decoder_data_center_well['x00']+decoder_data_center_well['x01']+
                                               decoder_data_center_well['x02']+decoder_data_center_well['x03']+
                                                decoder_data_center_well['x04']+decoder_data_center_well['x05']+
                                               decoder_data_center_well['x06']+decoder_data_center_well['x07']+
                                               decoder_data_center_well['x08'])

        task2_decode_center = decoder_data_center_well[(decoder_data_center_well['center_well_dist']<well_distance_max)&
                                (decoder_data_center_well['taskState']==2)]


        key['reward_times'] = unix_sound_cue
        key['reward_times_2'] = unix_sound_cue_2
        key['offset_diff'] = np.diff(time_offset)
        key['offset'] = average_time_offset

        self.insert1(key)
