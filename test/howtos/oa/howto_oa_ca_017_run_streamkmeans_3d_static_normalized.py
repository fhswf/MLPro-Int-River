## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_int_river
## -- Module  : howto_oa_ca_017_run_streamkmeans_3d_static_normalized.py
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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.2 (2024-01-05)

This module demonstrates the combination of several tasks in a workflow, which includes:

1) Boundary Detector,

2) Min/Max-Normalizer, and

3) Wrapped CluStream Algorithm (River).

Two data stream are incorporated in this module, such as static 3D point clouds and dynamic 3D point
clouds. In this module, we demonstrate the workflow in static 3D point clouds.

This module is prepared for the MLPro-OA scientific paper and going to be stored as Code
Ocean Capsule, thus the result is reproducible.

"""


from mlpro.bf.streams.streams import *
from mlpro.bf.streams.streams.clouds3d_static import StreamMLProStaticClouds3D
from mlpro.bf.various import Log

from mlpro.oa.streams import *
from mlpro_int_river.wrappers.clusteranalyzers import *





# 0 Prepare Demo/Unit test mode
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





# 1 Prepare a scenario for Static 3D Point Clouds
class Static3DScenario(OAScenario):

    C_NAME = 'Static3DScenario'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1.1 Get stream from StreamMLProStaticClouds3D
        stream = StreamMLProClouds( p_num_dim = 3,
                                    p_num_instances = 2000,
                                    p_num_clouds = 5,
                                    p_seed = 1,
                                    p_radii = [100, 150, 200, 250, 300],
                                    p_weights = [2,3,4,5,6],
                                    p_logging=Log.C_LOG_NOTHING )

        # 1.2 Set up a stream workflow based on a custom stream task

        # 1.2.1 Creation of a workflow
        workflow = OAWorkflow(p_name='wf_3D',
                              p_range_max=OAWorkflow.C_RANGE_NONE,
                              p_ada=p_ada,
                              p_visualize=p_visualize,
                              p_logging=p_logging)


        # 1.2.2 Creation of tasks and add them to the workflow

        # Boundary detector 
        task_bd = BoundaryDetector(p_name='t1', 
                                   p_ada=True, 
                                   p_visualize=p_visualize,   
                                   p_logging=p_logging)
        
        workflow.add_task(p_task = task_bd)

        # MinMax-Normalizer
        task_norm_minmax = NormalizerMinMax(p_name='t2', 
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





# 2 Instantiate the stream scenario
myscenario = Static3DScenario(
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

clusters            = myscenario.get_workflow()._tasks[2].get_clusters()
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