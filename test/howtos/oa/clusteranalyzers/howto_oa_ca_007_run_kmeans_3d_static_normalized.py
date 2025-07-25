## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_river
## -- Module  : howto_oa_ca_007_run_kmeans_3d_static_normalized.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-08-23  0.0.0     SY       Creation
## -- 2023-08-23  1.0.0     SY       First version release
## -- 2023-11-19  1.0.1     DA       Turned on visualization/logging of clustering task
## -- 2023-08-20  1.0.2     SY       Refactoring due to failed in Unittest
## -- 2023-12-10  1.0.3     DA       Increased number of macro clusters of CluStream to 8
## -- 2023-12-20  1.1.0     DA       Added event-oriented renormalization to CluStream task
## -- 2023-12-21  1.1.1     SY       Refactoring
## -- 2024-01-05  1.1.2     SY       Updating KMeans parameters
## -- 2024-04-30  1.2.0     DA       Alignment with MLPro 2
## -- 2024-05-27  1.2.1     SY       Printing clusters' sizes
## -- 2024-11-27  1.3.0     DA       Alignment with MLPro 2
## -- 2024-12-03  1.4.0     DA       Alignment with MLPro 2
## -- 2025-07-23  1.5.0     DA       Alignment with MLPro 2.1
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.5.0 (2025-07-23)

This module demonstrates online cluster analysis of normalized static 3D random point clouds using the wrapped
River implementation of stream algorithm KMeans. To this regard, the systematics of sub-framework 
MLPro-OA-Streams for online adaptive stream processing is used to implement a scenario consisting of  
a custom workflow and a native benchmark stream.

In particular you will learn:

1. How to set up, run and visualize an online adaptive custom stream processing scenario 

2. How to reuse wrapped River algorithms in own custom stream processing workflows

3. How to reuse native MLPro benchmark streams

4. How to reuse native MLPro online adaptive min-max normalization for data preprocessing

"""


from datetime import datetime

from mlpro.bf import Log, Mode, PlotSettings
from mlpro.bf.streams.streams.clouds import *
from mlpro.oa.streams import *
from mlpro.oa.streams.tasks import BoundaryDetector, NormalizerMinMax

from mlpro_int_river.wrappers.clusteranalyzers import WrRiverKMeans2MLPro



# 1 Prepare a scenario for Static 3D Point Clouds
class Static3DScenario(OAStreamScenario):

    C_NAME = 'Static3DScenario'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1.1 Get stream from StreamMLProClouds
        stream = StreamMLProClouds( p_num_dim = 3,
                                    p_num_instances = 2000,
                                    p_num_clouds = 5,
                                    p_seed = 1,
                                    p_radii = [100, 150, 200, 250, 300],
                                    p_weights = [2,3,4,5,6],
                                    p_logging=Log.C_LOG_NOTHING )

        # 1.2 Set up a stream workflow based on a custom stream task

        # 1.2.1 Creation of a workflow
        workflow = OAStreamWorkflow( p_name='Cluster Analysis using KMeans@River',
                                     p_range_max=OAStreamWorkflow.C_RANGE_NONE,
                                     p_ada=p_ada,
                                     p_visualize=p_visualize,
                                     p_logging=p_logging )


        # 1.2.2 Creation of tasks and add them to the workflow

        # Boundary detector 
        task_bd = BoundaryDetector(p_name='#1: Boundary Detector', 
                                   p_ada=True, 
                                   p_visualize=p_visualize,   
                                   p_logging=p_logging)
        
        workflow.add_task(p_task = task_bd)

        # MinMax-Normalizer
        task_norm_minmax = NormalizerMinMax(p_name='#2: Normalizer MinMax ', 
                                            p_ada=True,
                                            p_visualize=p_visualize, 
                                            p_logging=p_logging )

        task_bd.register_event_handler(
            p_event_id=BoundaryDetector.C_EVENT_ADAPTED,
            p_event_handler=task_norm_minmax.adapt_on_event
            )
        
        workflow.add_task(p_task = task_norm_minmax, p_pred_tasks=[task_bd])

        # Cluster Analyzer
        task_clusterer = WrRiverKMeans2MLPro( p_name='#3: KMeans@River',
                                              p_n_clusters=5,
                                              p_halflife=0.25, 
                                              p_sigma=1.5, 
                                              p_mu=0.0,
                                              p_seed=30,
                                              p_visualize=p_visualize,
                                              p_logging=p_logging )
        
        task_norm_minmax.register_event_handler( p_event_id=NormalizerMinMax.C_EVENT_ADAPTED,
                                                 p_event_handler=task_clusterer.renormalize_on_event )
        
        workflow.add_task(p_task = task_clusterer, p_pred_tasks=[task_norm_minmax])

        # 1.3 Return stream and workflow
        return stream, workflow



# 2 Prepare Demo/Unit test mode
if __name__ == '__main__':
    cycle_limit = 1200
    logging     = Log.C_LOG_ALL
    visualize   = True
    step_rate   = 1
else:
    cycle_limit = 2
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1



# 3 Instantiate the stream scenario
myscenario = Static3DScenario(
    p_mode=Mode.C_MODE_REAL,
    p_cycle_limit=cycle_limit,
    p_visualize=visualize,
    p_logging=logging)



# 4 Reset and run own stream scenario
myscenario.reset()

if __name__ == '__main__':
    myscenario.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                        p_step_rate = step_rate ) )
    input('\nPlease arrange all windows and press ENTER to start stream processing...')

tp_before           = datetime.now()
myscenario.run()
tp_after            = datetime.now()
tp_delta            = tp_after - tp_before
duraction_sec       = ( tp_delta.seconds * 1000000 + tp_delta.microseconds + 1 ) / 1000000
myscenario.log(Log.C_LOG_TYPE_S, 'Duration [sec]:', round(duraction_sec,2), ', Cycles/sec:', round(cycle_limit/duraction_sec,2))

clusters            = myscenario.get_workflow()._tasks[2].clusters
number_of_clusters  = len(clusters)

myscenario.log(Log.C_LOG_TYPE_I, '-------------------------------------------------------')
myscenario.log(Log.C_LOG_TYPE_I, '-------------------------------------------------------')
myscenario.log(Log.C_LOG_TYPE_I, 'Here is the recap of the cluster analyzer')
myscenario.log(Log.C_LOG_TYPE_I, 'Number of clusters: ', number_of_clusters)
for x in range(number_of_clusters):
    myscenario.log(Log.C_LOG_TYPE_I, 'Center of Cluster ', str(x+1), ': ', list(clusters[x].centroid.value))
    myscenario.log(Log.C_LOG_TYPE_I, 'Size of Cluster ', str(x+1), ': ', clusters[x].size.value)
myscenario.log(Log.C_LOG_TYPE_I, '-------------------------------------------------------')
myscenario.log(Log.C_LOG_TYPE_I, '-------------------------------------------------------')

if __name__ == '__main__':
    input('Press ENTER to exit...')