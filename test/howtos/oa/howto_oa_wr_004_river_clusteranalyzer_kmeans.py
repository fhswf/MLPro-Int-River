## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.examples
## -- Module  : howto_oa_wr_004_river_clusteranalyzer_kmeans.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-05  0.0.0     SY       Creation
## -- 2023-06-05  1.0.0     SY       First version release
## -- 2023-08-23  1.0.1     SY       Refactoring
## -- 2023-12-17  1.0.2     SY       Refactoring unit test mode
## -- 2023-12-22  1.0.3     SY       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.3 (2023-12-22)

This module demonstrates the principles of stream processing with MLPro. To this regard, a stream of
a stream provider is combined with a stream workflow to a stream scenario. The workflow consists of 
a standard task 'Cluster Analyzer'. The stream scenario is used to process some instances. Moreover,
we reuse a number of cluster analyzer algorithms from river package, which will also be demonstrated
in this howto file.

You will learn:

1) How to set up a stream workflow based on stream tasks.

2) How to set up a stream scenario based on a stream and a processing stream workflow.

3) How to add a task ClusterAnalyzer.

4) How to reuse a cluster analyzer algorithm from river (https://www.riverml.xyz/), specifically
KMeans

"""

from mlpro.bf.streams.streams import *
from mlpro.bf.streams.models import *
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase

from mlpro.oa.streams import *
from mlpro_int_river.wrappers.clusteranalyzers import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Stream4KMeans (StreamMLProBase):

    C_ID                = 'St4KMeans'
    C_NAME              = 'Stream4KMeans'
    C_VERSION           = '1.0.0'
    C_NUM_INSTANCES     = 6

    C_SCIREF_URL        = 'https://riverml.xyz/latest/api/cluster/KMeans/'


## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        
        feature_space : MSpace = MSpace()

        for i in range(2):
            feature_space.add_dim( Feature( p_name_short = 'f' + str(i),
                                            p_base_set = Feature.C_BASE_SET_R,
                                            p_name_long = 'Feature #' + str(i),
                                            p_name_latex = '',
                                            p_description = '',
                                            p_symmetrical = False,
                                            p_logging=Log.C_LOG_NOTHING ) )

        return feature_space


## -------------------------------------------------------------------------------------------------
    def _init_dataset(self):

        # Prepare a test dataset from https://riverml.xyz/latest/api/cluster/KMeans/
        
        X = [
            [1, 2],
            [1, 4],
            [1, 0],
            [-4, 2],
            [-4, 4],
            [-4, 0]
            ]

        self._dataset   = np.array(X)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AdScenario4KMeans (OAScenario):

    C_NAME = 'AdScenario4KMeans'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1 Get stream from StreamKMeans
        stream = Stream4KMeans( p_logging=0 )

        # 2 Set up a stream workflow based on a custom stream task

        # 2.1 Creation of a workflow
        workflow = OAWorkflow( p_name='wf1',
                               p_range_max=OAWorkflow.C_RANGE_NONE,
                               p_ada=p_ada,
                               p_visualize=p_visualize, 
                               p_logging=p_logging )


        # 2.2 Creation of a cluster analzer task
        clusterer = WrRiverKMeans2MLPro( p_name='t1',
                                        p_n_clusters=2,
                                        p_halflife=0.1, 
                                        p_sigma=3, 
                                        p_seed=42,
                                        p_visualize=p_visualize, 
                                        p_logging=p_logging )

        workflow.add_task( p_task=clusterer )

        # 3 Return stream and workflow
        return stream, workflow





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 1 Preparation of demo/unit test mode
if __name__ == "__main__":
    # 1.1 Parameters for demo mode
    logging     = Log.C_LOG_ALL
    visualize   = True
    cycle_limit = 6
    step_rate   = 1

else:
    # 1.2 Parameters for internal unit test
    cycle_limit = 6
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1



# 2 Instantiate the stream scenario
myscenario = AdScenario4KMeans( p_mode=Mode.C_MODE_REAL,
                               p_cycle_limit=cycle_limit,
                               p_visualize=visualize,
                               p_logging=logging )



# 3 Reset and run own stream scenario
myscenario.reset()

if __name__ == '__main__':
    myscenario.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                        p_step_rate = step_rate ) )
    input('\nPlease arrange all windows and press ENTER to start stream processing...')

tp_before = datetime.now()
myscenario.run()
tp_after = datetime.now()
tp_delta = tp_after - tp_before
duraction_sec = ( tp_delta.seconds * 1000000 + tp_delta.microseconds + 1 ) / 1000000
myscenario.log(Log.C_LOG_TYPE_S, 'Duration [sec]:', round(duraction_sec,2), ', Cycles/sec:', round(cycle_limit/duraction_sec,2))



# 4 Validating the number of clusters and centers of each cluster between original algorithm and wrapper
wr_n_clusters       = len(myscenario.get_workflow()._tasks[0].get_clusters())

if wr_n_clusters == 2:
    print("The number of clusters from river and mlpro matches!")
else:
    print("The number of clusters from river and mlpro does not match!")
    
    
river_centers       = myscenario.get_workflow()._tasks[0].get_algorithm().centers

for x in range(wr_n_clusters):
    if list(river_centers[x].values()) == list(myscenario.get_workflow()._tasks[0].get_clusters()[x].get_centroid().get_values()):
        print("The center of cluster %s from river and mlpro matches!"%(x+1))
    else:
        print("The center of cluster %s from river and mlpro does not match!"%(x+1))



if __name__ == '__main__':
    input('Press ENTER to exit...')