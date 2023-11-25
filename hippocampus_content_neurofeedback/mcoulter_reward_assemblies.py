# required packages:
import datajoint as dj
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

from hippocampus_content_neurofeedback.mcoulter_reward_cluster_list import RewardClusterList
from hippocampus_content_neurofeedback.mcoulter_realtime_filename import RealtimeFilename
from spyglass.decoding import SortedSpikesIndicator
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

schema = dj.schema("mcoulter_reward_assemblies")


# parameters
@schema
class CellAssemblyParameters(dj.Manual):
    definition = """
    assembly_param_name : varchar(500)
    ---
    assembly_parameters : blob
    """

    # can include description above
    # description : varchar(500)

    def insert_default(self):
        assembly_parameters = {}
        assembly_parameters["cluster_type"] = 'new_100ms'
        assembly_parameters["sum_bins"] = 5
        assembly_parameters["nshu"] = 100
        assembly_parameters["percentile"] = 99.5

        self.insert1(
            ["default", assembly_parameters], skip_duplicates=True
        )


# i dont think we need samplecount or statescriptfile in this list
# select parameters and cluster
@schema
class CellAssemblySelection(dj.Manual):
    definition = """
    -> RealtimeFilename
    -> CellAssemblyParameters
    ---
    """


# this is the computed table - basically it has the results of the analysis
@schema
class CellAssembly(dj.Computed):
    definition = """
    -> CellAssemblySelection
    ---
    assembly_dictionary : mediumblob

    """

    def make(self, key):
        print(f"Computing cell asssembly for: {key}")

        assembly_parameters = (
            CellAssemblyParameters
            & {
                "assembly_param_name": key[
                    "assembly_param_name"
                ]
            }
        ).fetch()[0][1]

        cluster_type = assembly_parameters["cluster_type"]
        sum_bins = assembly_parameters["sum_bins"]
        nshu = assembly_parameters["nshu"]
        percentile = assembly_parameters["percentile"]

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


        # new assembly dictionary
        assembly_dictionary = {}

        last_pos_interval = []
        try:
            reward_cluster_dict = (RewardClusterList() & {'nwb_file_name':key['nwb_file_name']} & 
                                {'reward_cluster_list_param_name' : cluster_type}).fetch('reward_cluster_dict')  
            assembly_dictionary[key['nwb_file_name']] = {}

            sort_interval_list = []
            for item in reward_cluster_dict[0][key['nwb_file_name']].items():
                #print(item[0])
                sort_interval_list.append(item[0])

            #for sort_interval in sort_interval_list:
            for sort_interval in sort_interval_list:  
                assembly_dictionary[key['nwb_file_name']][sort_interval] = {}
                
                print(sort_interval)
                pos_interval_list = []
                
                for item in reward_cluster_dict[0][key['nwb_file_name']][sort_interval].items():
                    #print(item[0])
                    pos_interval_list.append(item[0])        

                for pos_interval in pos_interval_list:
                    all_units = []
                    assembly_dictionary[key['nwb_file_name']][sort_interval][pos_interval] = {}
                    reward_cluster_out_list = []
                    print(pos_interval)
                    if pos_interval == last_pos_interval:
                        print('same interval, skip',last_pos_interval,pos_interval)
                    else:
                        spike_only_dict = {}
                        cluster_counter = 0
                        unit_counter = 0

                        # skip if no spikes
                        if len(reward_cluster_dict[0][key['nwb_file_name']][sort_interval][pos_interval].items()) > 0:

                            for item in reward_cluster_dict[0][key['nwb_file_name']][sort_interval][pos_interval].items():
                                print('reward cluster',item[0].split('_')[1],item[0].split('_')[2])
                                reward_cluster_out_list.append([np.int(item[0].split('_')[1]),
                                                                np.int(item[0].split('_')[2])])
                                #print('reward spikes',item[1][6])
                                spike_only_dict[cluster_counter] = item[1][6]
                                #print('spike count',len(spike_only_dict[cluster_counter]))
                                cluster_counter += 1
                                if np.int(item[0].split('_')[1]) < 10 and np.int(item[0].split('_')[2]) < 10:
                                    column_name = str('000')+str(item[0].split('_')[1])+'_'+str('000')+str(item[0].split('_')[2])
                                elif np.int(item[0].split('_')[1]) < 10:
                                    column_name = str('000')+str(item[0].split('_')[1])+'_'+str('00')+str(item[0].split('_')[2])
                                elif np.int(item[0].split('_')[2]) < 10:
                                    column_name = str('00')+str(item[0].split('_')[1])+'_'+str('000')+str(item[0].split('_')[2])
                                else:
                                    column_name = str('00')+str(item[0].split('_')[1])+'_'+str('00')+str(item[0].split('_')[2])
                                #column_name_list.append(column_name)
                                
                                try:
                                    tet_units = np.array((SortedSpikesIndicator & {'nwb_file_name' : key['nwb_file_name']} & 
                                        {'sort_interval_name':sort_interval} &                
                                        {'interval_list_name':pos_interval} & {'sort_group_id':item[0].split('_')[1]} &
                                    {'artifact_removed_interval_list_name LIKE "%ampl_100_%"'}).fetch_dataframe()[column_name])
                                    #print(tet_units.shape)
                                    if unit_counter > 0:
                                        all_units = np.vstack((all_units, tet_units))
                                    elif unit_counter == 0:
                                        all_units = tet_units.copy()
                                    unit_counter += 1
                                except ValueError as e:
                                    print('no sorted spikes')
                                
                            if len(all_units) > 0:
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
                                assembly_dictionary[key['nwb_file_name']][sort_interval][pos_interval] = (reward_cluster_out_list,
                                                                                                assemblyAct)
                        else:
                            print('no spikes')
        except IndexError as e:
            print('no cluster data:',key['nwb_file_name'])

        key["assembly_dictionary"] = assembly_dictionary

        self.insert1(key)