# required packages:
import datajoint as dj
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

from hippocampus_content_neurofeedback.mcoulter_reward_cluster_list import RewardClusterList
from hippocampus_content_neurofeedback.mcoulter_realtime_filename import RealtimeFilename
from hippocampus_content_neurofeedback.mcoulter_arm_end_clusters import ArmEndClusters
from spyglass.decoding.v0.sorted_spikes import SortedSpikesIndicator
from spyglass.common import IntervalList
from spyglass.spikesorting.v0 import Waveforms, CuratedSpikeSorting
from spyglass.mcoulter_statescript_rewards import (StatescriptReward)
from spyglass.utils.nwb_helper_fn import close_nwb_files

import pprint
import warnings
# import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import logging
import datetime
import statistics
import os

from sklearn.decomposition import PCA
from scipy import stats
from numpy import matlib as mb
np.matlib = mb

FORMAT = "%(asctime)s %(message)s"

logging.basicConfig(level="INFO", format=FORMAT, datefmt="%d-%b-%y %H:%M:%S")
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

schema = dj.schema("mcoulter_arm_end_assemblies")


# parameters
@schema
class ArmEndCellAssemblyParameters(dj.Manual):
    definition = """
    arm_end_assembly_param_name : varchar(500)
    ---
    arm_end_assembly_parameters : blob
    """

    # can include description above
    # description : varchar(500)

    def insert_default(self):
        arm_end_assembly_parameters = {}
        arm_end_assembly_parameters["sum_bins"] = 5
        arm_end_assembly_parameters["nshu"] = 100
        arm_end_assembly_parameters["percentile"] = 99.5
        arm_end_assembly_parameters["min_spikes"] = 100
        arm_end_assembly_parameters["interneuron_FR"] = 7

        self.insert1(
            ["default", arm_end_assembly_parameters], skip_duplicates=True
        )


# i dont think we need samplecount or statescriptfile in this list
# select parameters and cluster
@schema
class ArmEndCellAssemblySelection(dj.Manual):
    definition = """
    -> RealtimeFilename
    -> ArmEndCellAssemblyParameters
    ---
    """


# this is the computed table - basically it has the results of the analysis
@schema
class ArmEndCellAssembly(dj.Computed):
    definition = """
    -> ArmEndCellAssemblySelection
    ---
    input_cluster_count : double
    assembly_dictionary : mediumblob

    """

    def make(self, key):
        print(f"Computing cell asssembly for: {key}")
        
        # close any open files
        close_nwb_files()

        assembly_parameters = (
            ArmEndCellAssemblyParameters
            & {
                "arm_end_assembly_param_name": key[
                    "arm_end_assembly_param_name"
                ]
            }
        ).fetch()[0][1]

        sum_bins = assembly_parameters["sum_bins"]
        nshu = assembly_parameters["nshu"]
        percentile = assembly_parameters["percentile"]
        min_spikes = assembly_parameters["min_spikes"]
        interneuron_FR = assembly_parameters["interneuron_FR"]
        all_clusters = assembly_parameters["all_clusters"]

        #__author__ = "Vítor Lopes dos Santos"
        #__version__ = "2019.1"

        def marcenkopastur(significance):
            
                nbins = significance.nbins
                nneurons = significance.nneurons
                tracywidom = significance.tracywidom
            
                # calculates statistical threshold from Marcenko-Pastur distribution
                q = float(nbins)/float(nneurons) # note that silent neurons are counted too
                lambdaMax = pow((1+np.sqrt(1/q)),2)
                lambdaMax += tracywidom*pow(nneurons,-2./3) # Tracy-Widom correction 
                
                return lambdaMax
            
        def getlambdacontrol(zactmat_):

                significance_ = PCA()
                significance_.fit(zactmat_.T)
                lambdamax_ = np.max(significance_.explained_variance_)
                
                return lambdamax_
            
        def binshuffling(zactmat,significance):
            
                np.random.seed()

                lambdamax_ = np.zeros(significance.nshu)
                for shui in range(significance.nshu):
                        zactmat_ = np.copy(zactmat)
                        for (neuroni,activity) in enumerate(zactmat_):
                                randomorder = np.argsort(np.random.rand(significance.nbins))
                                zactmat_[neuroni,:] = activity[randomorder]
                        lambdamax_[shui] = getlambdacontrol(zactmat_)

                lambdaMax = np.percentile(lambdamax_,significance.percentile)
                
                return lambdaMax
            
        def circshuffling(zactmat,significance):
            
                np.random.seed()

                lambdamax_ = np.zeros(significance.nshu)
                for shui in range(significance.nshu):
                        zactmat_ = np.copy(zactmat)
                        for (neuroni,activity) in enumerate(zactmat_):
                                cut = int(np.random.randint(significance.nbins*2))
                                zactmat_[neuroni,:] = np.roll(activity,cut)
                        lambdamax_[shui] = getlambdacontrol(zactmat_)

                lambdaMax = np.percentile(lambdamax_,significance.percentile)
                
                return lambdaMax

        def runSignificance(zactmat,significance):
            
                if significance.nullhyp == 'mp':
                        lambdaMax = marcenkopastur(significance)
                elif significance.nullhyp == 'bin':
                        lambdaMax = binshuffling(zactmat,significance)
                elif significance.nullhyp == 'circ':
                        lambdaMax = circshuffling(zactmat,significance)
                else: 
                        print('ERROR !')
                        print('    nyll hypothesis method '+str(nullhyp)+' not understood')
                        significance.nassemblies = np.nan
                        
                nassemblies = np.sum(significance.explained_variance_>lambdaMax)
                significance.nassemblies = nassemblies
                significance.lambdaMax = lambdaMax
                
                return significance
                
        def extractPatterns(actmat,significance,method):
                nassemblies = significance.nassemblies
            
                if method == 'pca':
                        idxs = np.argsort(-significance.explained_variance_)[0:nassemblies]
                        patterns = significance.components_[idxs,:]
                elif method == 'ica':
                        from sklearn.decomposition import FastICA
                        ica = FastICA(n_components=nassemblies)
                        ica.fit(actmat.T)
                        patterns = ica.components_
                else:
                        print('ERROR !')
                        print('    assembly extraction method '+str(method)+' not understood')
                        patterns = np.nan
                        
                
                        
                if patterns is not np.nan:
                    
                        patterns = patterns.reshape(nassemblies,-1)
                        
                        # sets norm of assembly vectors to 1
                        norms = np.linalg.norm(patterns,axis=1)
                        patterns /= np.matlib.repmat(norms,np.size(patterns,1),1).T
                
                return patterns

        def runPatterns(actmat, method='ica', nullhyp = 'mp', nshu = 1000, percentile = 99, tracywidom = False):
            
                '''
                INPUTS
                
                    actmat:     activity matrix - numpy array (neurons, time bins) 
                    nullhyp:    defines how to generate statistical threshold for assembly detection.
                                    'bin' - bin shuffling, will shuffle time bins of each neuron independently
                                    'circ' - circular shuffling, will shift time bins of each neuron independently
                                                                        obs: mantains (virtually) autocorrelations
                                    'mp' - Marcenko-Pastur distribution - analytical threshold
                    nshu:       defines how many shuffling controls will be done (n/a if nullhyp is 'mp')
                    percentile: defines which percentile to be used use when shuffling methods are employed.
                                                                                (n/a if nullhyp is 'mp')                  
                    tracywidow: determines if Tracy-Widom is used. See Peyrache et al 2010.
                                                            (n/a if nullhyp is NOT 'mp')                                            
                OUTPUTS
                    patterns:     co-activation patterns (assemblies) - numpy array (assemblies, neurons)
                    significance: object containing general information about significance tests 
                    zactmat:      returns z-scored actmat
                '''
            
                nneurons = np.size(actmat,0)
                nbins = np.size(actmat,1)
                
                silentneurons = np.var(actmat,axis=1)==0
                actmat_ = actmat[~silentneurons,:]
                
                # z-scoring activity matrix
                zactmat_ = stats.zscore(actmat_,axis=1)
                        
                # running significance (estimating number of assemblies)
                significance = PCA()
                significance.fit(zactmat_.T)
                significance.nneurons = nneurons
                significance.nbins = nbins
                significance.nshu = nshu
                significance.percentile = percentile
                significance.tracywidom = tracywidom
                significance.nullhyp = nullhyp
                significance = runSignificance(zactmat_,significance)
                if np.isnan(significance.nassemblies):
                        return

                if significance.nassemblies<1:
                        print('WARNING !')
                        print('    no assembly detecded!')
                        patterns = []
                else:
                        # extracting co-activation patterns
                        patterns_ = extractPatterns(zactmat_,significance,method)
                        if patterns_ is np.nan:
                            return
                
                # putting eventual silent neurons back (their assembly weights are defined as zero)
                        patterns = np.zeros((np.size(patterns_,0),nneurons))
                        patterns[:,~silentneurons] = patterns_
                zactmat = np.copy(actmat.astype(float))
                zactmat[~silentneurons,:] = zactmat_
                
                return patterns,significance,zactmat
            
        def computeAssemblyActivity(patterns,zactmat,zerodiag = True):

                nassemblies = len(patterns)
                nbins = np.size(zactmat,1)

                assemblyAct = np.zeros((nassemblies,nbins))
                for (assemblyi,pattern) in enumerate(patterns):
                        projMat = np.outer(pattern,pattern)
                        projMat -= zerodiag*np.diag(np.diag(projMat))
                        for bini in range(nbins):
                                assemblyAct[assemblyi,bini] = \
                                        np.dot(np.dot(zactmat[:,bini],projMat),zactmat[:,bini])
                                
                return assemblyAct

        def similaritymat(patternsX,patternsY=None,method='cosine',findpairs=False):
            
                '''
                INPUTS
                    patternsX:     co-activation patterns (assemblies) 
                                                - numpy array (assemblies, neurons)
                    patternsY:     co-activation patterns (assemblies) 
                                                - numpy array (assemblies, neurons)
                                                - if None, will compute similarity
                                                            of patternsX to itself
                    
                    method:        defines similarity measure method
                                                'cosine' - cosine similarity
                    findpairs:     maximizes main diagonal of the sim matrix to define pairs \
                                                            from patterns X and Y
                                                returns rowind,colind which can be used to reorder 
                                                            patterns X and Y to maximize the diagonal                                      
                OUTPUTS
                    simmat:        similarity matrix
                                                - array (assemblies from X, assemblies from Y)
                '''
                
                if method == 'cosine':
                        from sklearn.metrics.pairwise import cosine_similarity as getsim
                else:
                        print(method +' for similarity has not been implemented yet.')
                        return
                    
                inputs = {'X': patternsX, 'Y': patternsY}
                simmat = getsim(**inputs)
                
                if findpairs:
                    
                        def fillmissingidxs(ind,n):
                                missing = list(set(np.arange(n))-set(ind))
                                ind = np.array(list(ind)+missing)
                                return ind
                                
                    
                        import scipy.optimize as optimize
                        rowind,colind = optimize.linear_sum_assignment(-simmat)
                        
                        rowind = fillmissingidxs(rowind,np.size(simmat,0))
                        colind = fillmissingidxs(colind,np.size(simmat,1))
                        
                        return simmat,rowind,colind
                else:
                        return simmat

        # need to insert a copy of target arm dictionaries
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
        
        #  empty variables
        assembly_dictionary = {}
        filtered_arm_end_clusters = []

        # arm end clusters: get filtered list
        #interneuron_FR = 7
        #min_spikes = 100

        try:
                cluster_count_in = len((ArmEndClusters & {'realtime_filename':key['realtime_filename']}).fetch('cluster_id')[0])
        except IndexError as e:
                print('no arm clusters',key['nwb_file_name'])
                cluster_count_in = 0
                
        if cluster_count_in > 0:
        
                print(key['nwb_file_name'],len((ArmEndClusters & {'realtime_filename':key['realtime_filename']}).fetch('cluster_id')[0]))
                cluster_id_list = (ArmEndClusters & {'realtime_filename':key['realtime_filename']}).fetch('cluster_id')[0]
                arm1_spikes = (ArmEndClusters & {'realtime_filename':key['realtime_filename']}).fetch('arm1_end_spikes')[0]   
                arm2_spikes = (ArmEndClusters & {'realtime_filename':key['realtime_filename']}).fetch('arm2_end_spikes')[0]  

                assembly_dictionary[key['nwb_file_name']] = {}
                assembly_dictionary[key['nwb_file_name']][key['interval_list_name']] = {}
                
                # specify dictionary 
                rat_name = key["nwb_file_name"].split("2")[0]

                # how to get realtime filename - not needed
                if rat_name == 'ron':
                        dict_name = ron_all_dict
                elif rat_name == 'tonks':
                        dict_name = tonks_all_dict
                elif rat_name == 'arthur':
                        dict_name = arthur_all_dict
                elif rat_name == 'molly':
                        dict_name = molly_all_dict
                elif rat_name == 'ginny':
                        dict_name = ginny_all_dict
                elif rat_name == 'pippin':
                        dict_name = pippin_all_dict

                target_arm = dict_name[key['nwb_file_name']]

                filtered_arm_end_clusters = []
                
                # override this step to get all clusters
                if all_clusters == 1:
                        cluster_id_list_new = cluster_id_list.copy()
                # arm 1
                elif target_arm == 1:
                        cluster_id_list_new = cluster_id_list[arm1_spikes>30]
                # arm 2
                elif target_arm == 2:
                        cluster_id_list_new = cluster_id_list[arm2_spikes>30]
                print('clusters in:',cluster_id_list_new.shape[0])
                
                reward_cluster_out_array = np.zeros((cluster_id_list_new.shape[0],2))
                for i in np.arange(cluster_id_list_new.shape[0]):
                        reward_cluster_out_array[i,0] = np.int(cluster_id_list_new[i].split('_')[0])
                        reward_cluster_out_array[i,1] = np.int(cluster_id_list_new[i].split('_')[1])

                # start and end for each sesssion
                start_time = (
                IntervalList
                & {"nwb_file_name": key["nwb_file_name"]}
                & {"interval_list_name": key["interval_list_name"]}
                ).fetch("valid_times")[0][0][0]
                end_time = (
                IntervalList
                & {"nwb_file_name": key["nwb_file_name"]}
                & {"interval_list_name": key["interval_list_name"]}
                ).fetch("valid_times")[0][0][1]
                session_duration = end_time - start_time
                print('duration, mins:',session_duration/60)
                
                # replace this based on looking up sort intervals from waveforms
                sort_intervals = np.unique(
                (
                        Waveforms
                        & {"nwb_file_name": key["nwb_file_name"]}
                        & {"waveform_params_name": "default_whitened"}
                        & {'artifact_removed_interval_list_name LIKE "%100_prop%"'}
                ).fetch("sort_interval_name")
                )
                print("sort intervals", sort_intervals)

                pos_name = 'pos '+str(np.int(key['interval_list_name'].split('_')[0])-1)+' valid times'

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
        
                # define list for manually curated
                tet_list_3 = np.unique((CuratedSpikeSorting & {"nwb_file_name": key["nwb_file_name"]} &
                                {"sort_interval_name": sort_interval} & {'curation_id':3}).fetch('sort_group_id'))

                # loop through all the clusters that have > 30 target spikes
                for i in np.arange(reward_cluster_out_array.shape[0]):
                        # need to define cluster and tetrode based on this array
                        # need to figure out how to get sort_interval
                        tetrode = np.int(reward_cluster_out_array[i,0])
                        cluster = np.int(reward_cluster_out_array[i,1])
                        
                        try:
                                if tetrode in tet_list_3:
                                        curation_id = 3
                                        cluster_spike_times = (
                                                CuratedSpikeSorting
                                                & {"nwb_file_name": key["nwb_file_name"]}
                                                & {"sort_interval_name": sort_interval}
                                                & {'artifact_removed_interval_list_name LIKE "%100_prop%"'}
                                                & {"sorter": "mountainsort4"}
                                                & {"curation_id": 3}
                                                & {"sort_group_id": tetrode}
                                                ).fetch_nwb()[0]["units"]["spike_times"][cluster]
                                else:
                                        curation_id = 1
                                        cluster_spike_times = (
                                                CuratedSpikeSorting
                                                & {"nwb_file_name": key["nwb_file_name"]}
                                                & {"sort_interval_name": sort_interval}
                                                & {'artifact_removed_interval_list_name LIKE "%100_prop%"'}
                                                & {"sorter": "mountainsort4"}
                                                & {"curation_id": 1}
                                                & {"sort_group_id": tetrode}
                                                ).fetch_nwb()[0]["units"]["spike_times"][cluster]
                        except (KeyError, IndexError) as e:
                                print('noise cluster',key['nwb_file_name'],tetrode,cluster)
                                cluster_spike_times = np.array([0,0,0])

                        session_spikes = cluster_spike_times[
                                (cluster_spike_times > start_time)
                                & (cluster_spike_times < end_time)
                        ]
                        
                        whole_firing_rate = (session_spikes.shape[0] / session_duration)

                        # remove MUA units
                        try:
                                cluster_label = (
                                        CuratedSpikeSorting.Unit
                                        & {"nwb_file_name": key["nwb_file_name"]}
                                        & {"sort_interval_name": sort_interval}
                                        & {'artifact_removed_interval_list_name LIKE "%100_prop%"'}
                                        & {"sorter": "mountainsort4"}
                                        & {"curation_id": curation_id}
                                        & {"sort_group_id": tetrode}
                                        & {"unit_id": cluster}
                                ).fetch("label")[0]
                                if cluster_label == "mua":
                                        pass
                                        #print("mua")
                                elif cluster_label == "":
                                                
                                        if (whole_firing_rate < interneuron_FR
                                        and session_spikes.shape[0] > min_spikes ):
                                                # add cluster to list for assembly analysis
                                                filtered_arm_end_clusters.append([tetrode,cluster])
                                                #print(whole_firing_rate, session_spikes.shape[0],session_duration/60)
                        except IndexError as e:
                                print('noise cluster',tetrode,cluster)
                        
                print('clusters out',len(filtered_arm_end_clusters))

                # new assembly dictionary
                #assembly_dictionary = {}
                # this will use the 2nd half of the first pair to get the middle session

                spike_only_dict = {}
                cluster_counter = 0
                unit_counter = 0

                for item in filtered_arm_end_clusters:
                        #print('reward cluster',item[0],item[1])
                        #reward_cluster_out_list.append([np.int(item[0].split('_')[1]),
                        #                                np.int(item[0].split('_')[2])])
                        #print('reward spikes',item[1][6])
                        #spike_only_dict[cluster_counter] = item[1][6]
                        #print('spike count',len(spike_only_dict[cluster_counter]))
                        cluster_counter += 1
                        if np.int(item[0]) < 10 and np.int(item[1]) < 10:
                                column_name = str('000')+str(item[0])+'_'+str('000')+str(item[1])
                        elif np.int(item[0]) < 10:
                                column_name = str('000')+str(item[0])+'_'+str('00')+str(item[1])
                        elif np.int(item[1]) < 10:
                                column_name = str('00')+str(item[0])+'_'+str('000')+str(item[1])
                        else:
                                column_name = str('00')+str(item[0])+'_'+str('00')+str(item[1])
                        #column_name_list.append(column_name)
                        
                        #print(column_name)

                        try:
                                if item[0] in tet_list_3:
                                        #print('manually curated tetrode at indicator')
                                        tet_units = np.array((SortedSpikesIndicator & {'nwb_file_name' : key['nwb_file_name']} & 
                                                {'sort_interval_name':sort_interval} & {'curation_id':3} &          
                                                {'interval_list_name':pos_name} & {'sort_group_id':item[0]} &
                                        {'artifact_removed_interval_list_name LIKE "%ampl_100_%"'}).fetch_dataframe()[column_name])
                                else:
                                        tet_units = np.array((SortedSpikesIndicator & {'nwb_file_name' : key['nwb_file_name']} & 
                                                {'sort_interval_name':sort_interval} & {'curation_id':1} &          
                                                {'interval_list_name':pos_name} & {'sort_group_id':item[0]} &
                                        {'artifact_removed_interval_list_name LIKE "%ampl_100_%"'}).fetch_dataframe()[column_name])
                                #print(tet_units.shape)
                                if unit_counter > 0:
                                        all_units = np.vstack((all_units, tet_units))
                                elif unit_counter == 0:
                                        all_units = tet_units.copy()
                                unit_counter += 1
                        except ValueError as e:
                                print('no sorted spikes',item)
                        
                if len(all_units) > 0:
                        try:
                                # find assemblies - code is below
                                # sum into 10 msec bins
                                #sum_bins = 5
                                max_index = (all_units.shape[1]//sum_bins)*sum_bins
                                all_units = all_units[:,:max_index]
                                binned_10ms = all_units[:,:].reshape(-1, np.int((all_units.shape[1])/sum_bins), sum_bins).sum(axis=2)
                                print('number units',binned_10ms.shape)

                                #nshu = 1000 # defines number of controls to run 
                                #percentile = 99.5 # defines percentile for significance threshold

                                #extractPatterns(all_units,0.01,'pca')
                                #patterns,significance,zactmat = runPatterns(binned_10ms,nullhyp='mp')
                                # with circular shuffle - for strong auto-correlations
                                patterns,significance,zactmat = runPatterns(binned_10ms,nullhyp='circ',nshu=nshu,percentile=percentile)

                                assemblyAct = computeAssemblyActivity(patterns,zactmat)
                                # note that the zactmat could be from another session (like a sleep session for "replay" analysis)

                                print('significant assemblies',assemblyAct[:,:].T.shape[1])

                                # save to dictionary
                                assembly_dictionary[key['nwb_file_name']][key['interval_list_name']] = (filtered_arm_end_clusters,patterns,assemblyAct)

                        except (IndexError, ValueError) as e:
                                print('index error on max_index, or no spikes in binned')
                else:
                        print('no spikes')
        else:
                print('no cluster data',key['nwb_file_name'],[key['interval_list_name']])

        key["assembly_dictionary"] = assembly_dictionary
        key["input_cluster_count"] = len(filtered_arm_end_clusters)

        self.insert1(key)
