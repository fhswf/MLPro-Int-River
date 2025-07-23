## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_river.wrappers.clusteranalyzers
## -- Module  : clustream.py
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


This module provides a wrapper class for the CluStream algorithm provided by River.

Learn more:
https://www.riverml.xyz/
https://riverml.xyz/latest/api/cluster/CluStream/

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
__all__ = [ 'WrRiverCluStream2MLPro' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrRiverCluStream2MLPro (WrClusterAnalyzerRiver2MLPro):
    """
    This is the wrapper class for CluStream clusterer.
    
    According to https://riverml.xyz/latest/api/cluster/CluStream/ :
    The CluStream algorithm maintains statistical information about the data using micro-clusters.
    These micro-clusters are temporal extensions of cluster feature vectors. The micro-clusters are
    stored at snapshots in time following a pyramidal pattern. This pattern allows to recall summary
    statistics from different time horizons.

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
    p_n_macro_clusters : int
        The number of clusters (k) for the k-means algorithm. Default: 5.
    p_max_micro_clusters : int
        The maximum number of micro-clusters to use. Default: 100.
    p_micro_cluster_r_factor : int
        Multiplier for the micro-cluster radius. When deciding to add a new data point to a
        micro-cluster, the maximum boundary is defined as a factor of the micro_cluster_r_factor
        of the RMS deviation of the data points in the micro-cluster from the centroid. Default: 2.
    p_time_window : int
        If the current time is T and the time window is h, we only consider the data that arrived
        within the period (T-h,T). Default: 1000.
    p_time_gap : int
        An incremental k-means is applied on the current set of micro-clusters after each time_gap
        to form the final macro-cluster solution. Default: 100.
    p_seed : int
        Random seed used for generating initial centroid positions. Default: None.
    p_halflife : float
        Amount by which to move the cluster centers, a reasonable value if between 0 and 1.
        Default: 0.5.
    p_mu : float
        Mean of the normal distribution used to instantiate cluster positions. Default: 1.
    p_sigma : float
        Standard deviation of the normal distribution used to instantiate cluster positions.
        Default: 1.
    p_p : int
        Power parameter for the Minkowski metric. When p=1, this corresponds to the Manhattan
        distance, while p=2 corresponds to the Euclidean distance. Default: 2.
    p_kwargs : dict
        Further optional named parameters. 
        
    """

    C_TYPE          = 'River Cluster Analyzer CluStream'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = None,
                 p_range_max = MLTask.C_RANGE_THREAD,
                 p_ada:bool = True,
                 p_visualize:bool = False,
                 p_logging = Log.C_LOG_ALL,
                 p_n_macro_clusters:int = 5,
                 p_max_micro_clusters:int = 100,
                 p_micro_cluster_r_factor:int = 2,
                 p_time_window:int = 1000,
                 p_time_gap:int = 100,
                 p_seed:int = None,
                 p_halflife:float = 0.5,
                 p_mu:float = 1,
                 p_sigma:float = 1,
                 p_p:int = 2,
                 **p_kwargs):
        
        alg = cluster.CluStream(n_macro_clusters=p_n_macro_clusters,
                                max_micro_clusters=p_max_micro_clusters,
                                micro_cluster_r_factor=p_micro_cluster_r_factor,
                                time_window=p_time_window,
                                time_gap=p_time_gap,
                                seed=p_seed,
                                halflife=p_halflife,
                                mu=p_mu,
                                sigma=p_sigma,
                                p=p_p)

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
        
        updated_cls = self._river_algo.predict_one(input_data)
        
        for x in self._river_algo.centers.keys():
            related_cluster = self._clusters[x]
            related_cluster.centroid.value = list(self._river_algo.centers[x].values())
            
            if x == updated_cls:
                act_size = related_cluster.size.value
                if act_size is not None:
                    related_cluster.size.value = act_size+1
                else:
                    related_cluster.size.value = 1


## -------------------------------------------------------------------------------------------------
    def _get_clusters(self):
        """
        This method returns the current list of clusters.

        Returns
        -------
        dict_of_clusters : dict[Cluster]
            Current list of clusters.
            
        """
        
        for x in self._river_algo.centers.keys():
            try:
                related_cluster = self._clusters[x]
            except:
                related_cluster = self._cls_cluster(p_id=x, p_properties=self.C_CLUSTER_PROPERTIES, p_visualize=self.get_visualization())   

                if self.get_visualization():
                    related_cluster.init_plot(p_figure = self._figure, p_plot_settings=self._plot_settings)             

                related_cluster.centroid.value = list(self._river_algo.centers[x].values())
                self._add_cluster( p_cluster = related_cluster )

        return self._clusters


## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer:Normalizer):
        """
        Internal enormalization is hard to realize here without getting strong dependencies on
        internal implementation details. 
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    clusters = property( fget=_get_clusters )