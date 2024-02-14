## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_river
## -- Module  : howto_oa_ca_004_run_kmeans_2d_dynamic_normalized.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-08-23  0.0.0     SY       Creation
## -- 2023-08-23  1.0.0     SY       First version release
## -- 2023-11-19  1.0.1     DA       Turned on visualization/logging of clustering task
## -- 2023-08-20  1.0.2     SY       - Refactoring due to failed in Unittest
## --                                - Add window to the workflow
## -- 2024-02-02  1.1.0     SY       Parameters Optimization
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2024-02-02)

This module demonstrates online cluster analysis of normalized dynamic 2D random point clouds using the wrapped
River implementation of stream algorithm KMeans. To this regard, the systematics of sub-framework 
MLPro-OA-Streams for online adaptive stream processing is used to implement a scenario consisting of  
a custom workflow and a native benchmark stream.

In particular you will learn:

1. How to set up, run and visualize an online adaptive custom stream processing scenario 

2. How to reuse wrapped River algorithms in own custom stream processing workflows

3. How to reuse native MLPro benchmark streams

4. How to reuse native MLPro sliding window

5. How to reuse native MLPro online adaptive min-max normalization for data preprocessing

"""


from mlpro.bf.streams.streams import *
from mlpro.bf.various import Log
from mlpro.bf.streams.tasks.windows import Window

from mlpro.oa.streams import *
from mlpro_int_river.wrappers.clusteranalyzers import *





# 0 Prepare Demo/Unit test mode
if __name__ == '__main__':
    cycle_limit = 2000
    logging     = Log.C_LOG_ALL
    visualize   = True
    step_rate   = 2
else:
    cycle_limit = 2
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1





# 1 Prepare a scenario for Dynamic 2D Point Clouds
class Dynamic2DScenario(OAScenario):

    C_NAME = 'Dynamic2DScenario'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1.1 Get stream from StreamMLProClouds
        stream = StreamMLProClouds( p_num_dim = 2,
                                    p_num_instances = 2000,
                                    p_num_clouds = 5,
                                    p_seed = 1,
                                    p_radii=[100],
                                    p_velocity=1,
                                    p_logging=Log.C_LOG_NOTHING )

        # 1.2 Set up a stream workflow based on a custom stream task

        # 1.2.1 Creation of a workflow
        workflow = OAWorkflow(p_name='wf_2D',
                              p_range_max=OAWorkflow.C_RANGE_NONE,
                              p_ada=p_ada,
                              p_visualize=p_visualize,
                              p_logging=p_logging)


        # 1.2.2 Creation of tasks and add them to the workflow
      
        # Window
        task_window = Window(p_buffer_size=100, 
                             p_delay=False,
                             p_enable_statistics=True,
                             p_name='t1',
                             p_duplicate_data=True,
                             p_visualize=p_visualize,
                             p_logging=p_logging)
        workflow.add_task(p_task=task_window)

        # Boundary detector 
        task_bd = BoundaryDetector(p_name='t2', 
                                   p_ada=True, 
                                   p_visualize=p_visualize,   
                                   p_logging=p_logging)
        workflow.add_task(p_task=task_bd, p_pred_tasks=[task_window])

        # MinMax-Normalizer
        task_norm_minmax = NormalizerMinMax(p_name='t3', 
                                            p_ada=True,
                                            p_visualize=p_visualize, 
                                            p_logging=p_logging )

        task_bd.register_event_handler(
            p_event_id=BoundaryDetector.C_EVENT_ADAPTED,
            p_event_handler=task_norm_minmax.adapt_on_event
            )
        workflow.add_task(p_task = task_norm_minmax, p_pred_tasks=[task_bd])

        # Cluster Analyzer
        task_clusterer = WrRiverKMeans2MLPro( p_name='t3',
                                             p_n_clusters=5,
                                             p_halflife=0.1, 
                                             p_sigma=-0.75, 
                                             p_seed=48,
                                             p_visualize=p_visualize,
                                             p_logging=p_logging )
       
        workflow.add_task(p_task = task_clusterer, p_pred_tasks=[task_norm_minmax])

        # 1.3 Return stream and workflow
        return stream, workflow





# 2 Instantiate the stream scenario
myscenario = Dynamic2DScenario(
    p_mode=Mode.C_MODE_REAL,
    p_cycle_limit=cycle_limit,
    p_visualize=visualize,
    p_logging=logging
    )




# 3 Reset and run own stream scenario
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

clusters            = myscenario.get_workflow()._tasks[3].get_clusters()
number_of_clusters  = len(clusters)

myscenario.log(Log.C_LOG_TYPE_I, '-------------------------------------------------------')
myscenario.log(Log.C_LOG_TYPE_I, '-------------------------------------------------------')
myscenario.log(Log.C_LOG_TYPE_I, 'Here is the recap of the cluster analyzer')
myscenario.log(Log.C_LOG_TYPE_I, 'Number of clusters: ', number_of_clusters)
for x in range(number_of_clusters):
    myscenario.log(Log.C_LOG_TYPE_I, 'Center of Cluster ', str(x+1), ': ', list(clusters[x].get_centroid().get_values()))
myscenario.log(Log.C_LOG_TYPE_I, '-------------------------------------------------------')
myscenario.log(Log.C_LOG_TYPE_I, '-------------------------------------------------------')

if __name__ == '__main__':
    input('Press ENTER to exit...')