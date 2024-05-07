## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_river.wrappers.clusteranalyzers
## -- Module  : dbstream.py
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
## -- 2024-05-07  1.4.0     DA       Separated to own module
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.4.0 (2024-05-07)

This module provides a wrapper class for the DBStream algorithm provided by River.

Learn more:
https://www.riverml.xyz/
https://riverml.xyz/latest/api/cluster/DBSTREAM/

"""


from mlpro_int_river.wrappers.clusteranalyzers.basics import WrClusterAnalyzerRiver2MLPro
from mlpro.bf.math.normalizers import Normalizer
from mlpro.oa.streams.tasks.clusteranalyzers.clusters import Cluster, ClusterCentroid
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import *
from mlpro.bf.mt import Task as MLTask
from mlpro.bf.various import Log
from mlpro.bf.streams import *
from river import cluster




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrRiverDBStream2MLPro (WrClusterAnalyzerRiver2MLPro):
    """
    This is the wrapper class for DBSTREAM clusterer.
    
    According to https://riverml.xyz/latest/api/cluster/DBSTREAM/ :
    DBSTREAM is a clustering algorithm for evolving data streams.
    It is the first micro-cluster-based online clustering component that explicitely captures the
    density between micro-clusters via a shared density graph. The density information in the graph
    is then exploited for reclustering based on actual density between adjacent micro clusters.
    
    The algorithm is divided into two parts:
        1) Online micro-cluster maintenance (learning)
        2) Offline generation of macro clusters (clustering)

    Parameters
    ----------
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
    p_clustering_threshold : float
        DBStream represents each micro cluster by a leader (a data point defining the micro cluster's
        center) and the density in an area of a user-specified radius (clustering_threshold) around
        the center. Default: 1.0.
    p_fading_factor : float
        Parameter that controls the importance of historical data to current cluster.
        Note that fading_factor has to be different from 0. Default: 0.01.
    p_cleanup_interval : float
        The time interval between two consecutive time points when the cleanup process is conducted.
        Default: 2.
    p_intersection_factor : float
        The intersection factor related to the area of the overlap of the micro clusters relative
        to the area cover by micro clusters. This parameter is used to determine whether a micro
        cluster or a shared density is weak. Default: 0.3.
    p_minimum_weight : float
        The minimum weight for a cluster to be not "noisy". Default: 1.0.
    p_kwargs : dict
        Further optional named parameters. 
        
    """

    C_TYPE                  = 'River Cluster Analyzer DBSTREAM'

    C_CLUSTER_PROPERTIES    = [ cprop_centroid ]

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = None,
                 p_range_max = MLTask.C_RANGE_THREAD,
                 p_ada:bool = True,
                 p_visualize:bool = False,
                 p_logging = Log.C_LOG_ALL,
                 p_clustering_threshold:float = 1.0,
                 p_fading_factor:float = 0.01,
                 p_cleanup_interval:float = 2,
                 p_intersection_factor:float = 0.3,
                 p_minimum_weight:float = 1.0,
                 **p_kwargs):
        
        alg = cluster.DBSTREAM(clustering_threshold=p_clustering_threshold,
                               fading_factor=p_fading_factor,
                               cleanup_interval=p_cleanup_interval,
                               intersection_factor=p_intersection_factor,
                               minimum_weight=p_minimum_weight)

        super().__init__(p_cls_cluster=ClusterCentroid,
                         p_river_algo=alg,
                         p_name=p_name,
                         p_range_max=p_range_max,
                         p_ada=p_ada,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _update_clusters(self):
        """
        This method is to update the centroids of each introduced cluster.
        """
        
        for _, (key, val) in enumerate(self._river_algo.micro_clusters.items()):
            related_cluster = self._clusters[id(val)]
            related_cluster.centroid.value = list(self._river_algo.centers[key].values())


## -------------------------------------------------------------------------------------------------
    def get_clusters(self) -> dict[Cluster]:
        """
        This method returns the current list of clusters.

        Returns
        -------
        dict_of_clusters : dict[Cluster]
            Current list of clusters.
            
        """

        list_keys_river = []
        for _, (key, val) in enumerate(self._river_algo.micro_clusters.items()):
            list_keys_river.append(id(val))
            try:
                related_cluster = self._clusters[id(val)]
            except:
                related_cluster = ClusterCentroid(
                    p_id = id(val),
                    p_properties=self.C_CLUSTER_PROPERTIES,
                    p_visualize=self.get_visualization() )
                    # p_cluster=self._river_algo.clusters[key],
                    # p_micro_cluster=val)
                
                if self.get_visualization():
                    related_cluster.init_plot(p_figure = self._figure, p_plot_settings=self._plot_settings)             

                related_cluster.centroid.value = list(self._river_algo.centers[key].values())
                self._add_cluster( p_cluster = related_cluster )
        
        list_keys_mlpro = list(self._clusters.keys())
        for x in list_keys_mlpro:
            if x not in list_keys_river:
                self._remove_cluster(self._clusters[x])

        return self._clusters


## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer:Normalizer):
        """
        Internal renormalization is hard to realize here without getting strong dependencies on
        internal implementation details. 
        """
        
        raise NotImplementedError
