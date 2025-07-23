## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_river.wrappers.clusteranalyzers
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-05-12  0.0.0     DA       Creation
## -- 2023-05-23  1.0.0     SY       First version release
## -- 2023-05-25  1.0.1     SY       Refactoring related to ClusterCentroid
## -- 2023-06-03  1.0.2     DA       Renaming of method ClusterAnalyzer.get_cluster_memberships
## -- 2023-06-05  1.0.3     SY       Updating get_cluster_memberships, p_cls_cluster, and _adapt
## -- 2023-11-20  1.0.4     SY       Update due to intorduction of visualization for ClusterCentroid
## -- 2023-12-08  1.0.5     SY       Update due to new data type in ClusterCentroid
## -- 2023-12-15  1.0.6     SY/DA    Adjustments on classes WrClusterAnalyzerRiver2MLPro and 
## --                                WrRiverClustStream2MLPro
## -- 2023-12-17  1.0.7     SY       Updates on classes WrRiverKMeans2MLPro, WrRiverDBStream2MLPro
## --                                WrRiverStreamKMeans2MLPro, WrRiverDenStream2MLPro
## -- 2023-12-21  1.0.8     SY       Add renormalization method on all wrapped algorithms
## -- 2023-12-22  1.0.9     DA/SY    Bugfix 
## -- 2023-12-25  1.0.10    DA       Bugfix in WrClusterAnalyzerRiver2MLPro._adapt()
## -- 2023-12-29  1.0.11    DA/SY    Disabled renormalization of CluStream, DBStream, DenStream
## -- 2024-01-05  1.0.12    DA       All classed: refactoring of naming (C_NAME)
## -- 2024-02-04  1.1.0     SY       Updating WrRiverDenStream2MLPro, WrRiverStreamKMeans2MLPro due
## --                                to visualization errors
## -- 2024-02-24  1.1.1     DA       Class WrClusterAnalyzerRiver2MLPro: package constants removed
## -- 2024-04-29  1.1.2     SY       Updating WrRiverDenStream2MLPro due to River 0.21.1
## -- 2024-04-30  1.2.0     DA       Alignment with MLPro 2
## -- 2024-05-05  1.3.0     DA       Alignment with MLPro 2
## -- 2024-05-07  1.4.0     DA       Separation of particular algorithms into separate modules
## -- 2024-05-24  1.5.0     DA       Alignment with MLPro 2
## -- 2024-05-25  1.5.1     SY       Introduction of size as a property, Refactoring
## -- 2024-07-04  1.6.0     DA       Alignment with MLPro 2
## -- 2025-06-15  1.7.0     DA       Alignment with MLpro 2.0.2
## -- 2025-07-21  1.7.1     DS       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.7.1 (2025-07-21)

This module provides wrapper root classes from River to MLPro, specifically for cluster analyzers. 

Learn more:
https://www.riverml.xyz/

"""


from typing import List, Tuple
import numpy as np

from mlpro.bf.streams import Instance
from mlpro_int_river.wrappers import WrapperRiver
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer
from mlpro.oa.streams.tasks.clusteranalyzers.clusters import Cluster
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import *
from mlpro.bf.mt import Task as MLTask
from mlpro.bf.various import Log
from mlpro.bf.streams import *

from river import base



# Export list for public API
__all__ = [ 'WrClusterAnalyzerRiver2MLPro' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrClusterAnalyzerRiver2MLPro (WrapperRiver, ClusterAnalyzer):
    """
    This is the base wrapper class for each River-based cluster analyzer to MLPro.

    Parameters
    ----------
    p_cls_cluster 
        Cluster class (Class Cluster or a child class).
    p_river_algo : river.base.Clusterer
        Instantiated river-based clusterer.
    p_name : str
        Name of the clusterer. Default: None.
    p_range_max :
        MLPro machine learning task, either process or thread. Default: MLTask.C_RANGE_THREAD.
    p_ada : bool
        Turn on adaptivity. Default: True.
    p_visualize : bool
        Turn on visualization. Default: False.
    p_logging :
        Set up type of logging. Default: Log.C_LOG_ALL.
    p_kwargs : dict
        Further optional named parameters. 
        
    """

    C_TYPE                  = 'River Cluster Analyzer'

    C_CLUSTER_PROPERTIES    = [ cprop_centroid, cprop_size ]

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_cls_cluster : type,
                  p_river_algo:base.Clusterer,
                  p_name:str = None,
                  p_range_max = MLTask.C_RANGE_THREAD,
                  p_ada:bool = True,
                  p_visualize:bool = True,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs):
        
        self._river_algo = p_river_algo

        ClusterAnalyzer.__init__( self,
                                  p_cls_cluster = p_cls_cluster,
                                  p_cluster_limit = 0,
                                  p_name = p_name,
                                  p_range_max = p_range_max,
                                  p_ada = p_ada,
                                  p_duplicate_data = False,
                                  p_visualize = p_visualize,
                                  p_logging = p_logging,
                                  **p_kwargs )

        WrapperRiver.__init__(self, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_instance_new : Instance) -> bool:
        """
        This method is to adapt the current clusters according to the incoming instances.

        Parameters
        ----------
        p_instance_new : Instance
            New stream instances to be processed.

        Returns
        -------
        bool
            True, if something has been adapted. False otherwise.        
        """
        
        # extract features data from instances
        feature_data    = np.array(p_instance_new.get_feature_data().get_values())

        # transform np array to dict with enumeration
        input_data = dict(enumerate(feature_data.flatten(), 1))

        # update the model with a set of features
        self.log(self.C_LOG_TYPE_I, 'Cluster is adapted...')
        self._river_algo.learn_one(input_data)
        
        # update MLPro clusters from river
        self._get_clusters()
        self._update_clusters(input_data)

        return True
    

## -------------------------------------------------------------------------------------------------
    def _update_clusters(self, input_data):
        """
        This method is to update the centroids of each introduced cluster.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_algorithm(self) -> base.Clusterer:
        """
        This method returns the river algorithm of the clusterer.

        Returns
        -------
        base.Clusterer
            The river algorithm of the clusterer.
            
        """

        return self._river_algo


## -------------------------------------------------------------------------------------------------
    def get_cluster_memberships( self, 
                                 p_instance: Instance, 
                                 p_scope: int = ClusterAnalyzer.C_RESULT_SCOPE_MAX ) -> List[Tuple[str, float, Cluster]]:
        """
        Public custom method to determine the membership of the given instance to each cluster as
        a value in percent.

        Parameters
        ----------
        p_instance : Instance
            Instance to be evaluated.
        p_scope : int
            Scope of the result list. See class attributes C_MS_SCOPE_* for possible values. Default
            value is C_MS_SCOPE_MAX.

        Returns
        -------
        membership : List[Tuple[str, float, Cluster]]
            List of membership tuples for each cluster. A tuple consists of a cluster id, a
            relative membership value in percent and a reference to the cluster.
            
        """
        
        # extract features data from instances
        feature_data    = p_instance.get_feature_data().get_values()

        # transform np array to dict with enumeration
        input_data = dict(enumerate(feature_data.flatten(), 1))

        # predict the cluster number according to a set of features
        cluster_idx = self._river_algo.predict_one(input_data)

        # get the corresponding cluster
        list_clusters = self.get_clusters().values()

        # return the cluster membership
        memberships_rel = []
        if list_clusters is not None:
            for x in range(len(list_clusters)):
                cluster = list_clusters[cluster_idx]
                if x == cluster_idx:
                    memberships_rel.append((cluster.id, 1, cluster))
                    self.log(self.C_LOG_TYPE_I,
                             'Actual instances belongs to cluster %s'%(cluster.id))
                else:
                    if p_scope == ClusterAnalyzer.C_MS_SCOPE_ALL:
                        memberships_rel.append((cluster.id, 0, cluster))

        return memberships_rel
