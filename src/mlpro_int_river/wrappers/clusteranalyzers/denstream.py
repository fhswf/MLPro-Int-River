## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_river.wrappers.clusteranalyzers
## -- Module  : denstream.py
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
## -- 2024-05-25  1.4.1     SY       Introduction of size as a property
## -- 2025-04-24  1.5.0     DA       Alignment with MLPro 2
## -- 2025-07-21  1.5.1     DS       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.5.1 (2025-07-21)

This module provides a wrapper class for the DenStream algorithm provided by River.

Learn more:
https://www.riverml.xyz/
https://riverml.xyz/latest/api/cluster/DenStream/

"""


from mlpro_int_river.wrappers.clusteranalyzers.basics import WrClusterAnalyzerRiver2MLPro
from mlpro.bf.math.normalizers import Normalizer
from mlpro.oa.streams.tasks.clusteranalyzers.clusters import Cluster, ClusterCentroid
from mlpro.oa.streams.tasks.clusteranalyzers.clusters.properties import *
from mlpro.bf.mt import Task as MLTask
from mlpro.bf.various import Log
from mlpro.bf.streams import *
from river import cluster



# Export list for public API
__all__ = [ 'WrRiverDenStream2MLPro' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrRiverDenStream2MLPro (WrClusterAnalyzerRiver2MLPro):
    """
    This is the wrapper class for DenStream clusterer.
    
    According to https://riverml.xyz/latest/api/cluster/DenStream/ :
    DenStream is a clustering algorithm for evolving data streams. DenStream can discover clusters
    with arbitrary shape and is robust against noise (outliers).

    "Dense" micro-clusters (named core-micro-clusters) summarise the clusters of arbitrary shape.
    A pruning strategy based on the concepts of potential and outlier micro-clusters guarantees the
    precision of the weights of the micro-clusters with limited memory.
    
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
    p_decaying_factor : float
        Parameter that controls the importance of historical data to current cluster. Note that
        decaying_factor has to be different from 0. Default: 0.25.
    p_beta : float
        Parameter to determine the threshold of outlier relative to core micro-clusters. The value
        of beta must be within the range (0,1). Default: 0.75.
    p_mu : float
        Parameter to determine the threshold of outliers relative to core micro-cluster.
        As beta * mu must be greater than 1, mu must be within the range (1/beta, inf). Default: 2.
    p_epsilon : float
        Defines the epsilon neighborhood. Default: 0.02.
    p_n_samples_init : int
        Number of points to to initiqalize the online process. Default: 1000.
    p_stream_speed : int
        Number of points arrived in unit time. Default: 100.
    p_kwargs : dict
        Further optional named parameters. 
        
    """

    C_TYPE          = 'River Cluster Analyzer DenStream'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = None,
                 p_range_max = MLTask.C_RANGE_THREAD,
                 p_ada:bool = True,
                 p_visualize:bool = False,
                 p_logging = Log.C_LOG_ALL,
                 p_decaying_factor:float = 0.25,
                 p_beta:float = 0.75,
                 p_mu:float = 2,
                 p_epsilon:float = 0.02,
                 p_n_samples_init:int = 1000,
                 p_stream_speed:int = 100,
                 **p_kwargs):
        
        alg = cluster.DenStream(decaying_factor=p_decaying_factor,
                                beta=p_beta,
                                mu=p_mu,
                                epsilon=p_epsilon,
                                n_samples_init=p_n_samples_init,
                                stream_speed=p_stream_speed)
        
        self.n_dummy_prediction = 0

        super().__init__(p_cls_cluster=ClusterCentroid,
                         p_river_algo=alg,
                         p_name=p_name,
                         p_range_max=p_range_max,
                         p_ada=p_ada,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _update_clusters(self, input_data):
        """
        This method is to update the centroids of each introduced cluster.
        """
        
        if self._river_algo.n_clusters != 0: 
            updated_cls = self._river_algo.predict_one(input_data)
            
            for val in self._river_algo.p_micro_clusters.values():
                related_cluster = self._clusters[id(val)]
                
                list_center = []
                for x, (_, val_center) in enumerate(val.x.items()):
                    list_center.append(val_center) 
                    
                try:
                    related_cluster.centroid.value = list_center        
                    if x == updated_cls:
                        act_size = related_cluster.size.value
                        if act_size is not None:
                            related_cluster.size.value = act_size+1
                        else:
                            related_cluster.size.value = 1
                except:
                    pass


## -------------------------------------------------------------------------------------------------
    def _get_clusters(self) -> dict[Cluster]:
        """
        This method returns the current list of clusters.

        Returns
        -------
        dict_of_clusters : dict[ClusterCentroid]
            Current list of clusters.
            
        """

        # dummy prediction
        if self.n_dummy_prediction == self._river_algo.n_samples_init+1:
            self._river_algo.predict_one({1: 0, 2: 0})
        else:
            self.n_dummy_prediction += 1

        list_keys_river = []
        if self._river_algo.n_clusters != 0:       
            for _, (key, val) in enumerate(self._river_algo.p_micro_clusters.items()):
                list_keys_river.append(id(val))
                try:
                    related_cluster = self._clusters[id(val)]
                except:
                    try:
                        o_micro_cluster = self._river_algo.o_micro_clusters[key]
                    except:
                        o_micro_cluster = None
                    related_cluster = self._cls_cluster(
                        p_id = id(val),
                        p_properties=self.C_CLUSTER_PROPERTIES,
                        p_visualize=self.get_visualization())
                        # p_cluster=self._river_algo.clusters[key],
                        # p_micro_cluster=val,
                        # p_o_micro_cluster=o_micro_cluster)
                    
                    if self.get_visualization():
                        related_cluster.init_plot(p_figure = self._figure, p_plot_settings=self._plot_settings)
    
                    list_center = []
                    for _, (_, val_center) in enumerate(val.x.items()):
                        list_center.append(val_center)           
                    related_cluster.centroid.value = list_center
                    self._add_cluster( p_cluster = related_cluster )
        
        list_keys_mlpro = list(self._clusters.keys())
        for x in list_keys_mlpro:
            if x not in list_keys_river:
                self._remove_cluster(self._clusters[x])

        return self._clusters


## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer:Normalizer):
        """
        Internal renormalization of all clusters. See method OATask.renormalize_on_event() for further
        information.

        Parameters
        ----------
        p_normalizer : Normalizer
            Normalizer object to be applied on task-specific 
        """
        
        super()._renormalize(p_normalizer)

        for cluster in self._clusters.values():
            for cluster_river in self._river_algo.p_micro_clusters.values():
                if id(cluster_river) == cluster.get_id():
                    for river_idx in cluster_river.x.keys():
                        cluster_river.x[river_idx] = cluster.centroid.value[river_idx-1]


## -------------------------------------------------------------------------------------------------
    clusters = property( fget=_get_clusters )