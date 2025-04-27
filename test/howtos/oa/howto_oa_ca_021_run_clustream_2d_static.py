## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_river
## -- Module  : howto_oa_ca_021_run_clustream_2d_static.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-02-09  0.0.0     SY       Creation
## -- 2024-02-09  1.0.0     SY       First version release
## -- 2024-04-30  1.1.0     DA       Alignment with MLPro 2
## -- 2024-05-27  1.1.1     SY       Printing clusters' sizes
## -- 2024-12-03  1.2.0     DA       Alignment with MLPro 2
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2024-12-03)

This module demonstrates online cluster analysis of static 2D random point clouds using the wrapped
River implementation of stream algorithm CluStream. To this regard, the systematics of sub-framework 
MLPro-OA-Streams for online adaptive stream processing is used to implement a scenario consisting of  
a custom workflow and a native benchmark stream.

In particular you will learn:

1. How to set up, run and visualize an online adaptive custom stream processing scenario 

2. How to reuse wrapped River algorithms in own custom stream processing workflows

3. How to reuse native MLPro benchmark streams

"""


from mlpro.bf.streams.streams import *
from mlpro.bf.streams.streams.clouds import *
from mlpro.bf.various import Log
from mlpro.oa.streams import *
from mlpro_int_river.wrappers.clusteranalyzers import WrRiverCluStream2MLPro



# 1 Prepare a scenario for Static 2D Point Clouds
class Static2DScenario(OAStreamScenario):

    C_NAME = 'Static2DScenario'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1.1 Get MLPro benchmark stream
        stream = StreamMLProClouds( p_num_dim = 2,
                                    p_num_instances = 2000,
                                    p_num_clouds = 5,
                                    p_seed = 1,
                                    p_radii=[100],
                                    p_logging=Log.C_LOG_NOTHING )

        # 1.2 Set up a stream workflow

        # 1.2.1 Creation of a workflow
        workflow = OAStreamWorkflow( p_name='Cluster Analysis using CluStream@River',
                                     p_range_max=OAStreamWorkflow.C_RANGE_NONE,
                                     p_ada=p_ada,
                                     p_visualize=p_visualize,
                                     p_logging=p_logging )


        # 1.2.2 Creation of tasks and add them to the workflow

        # Cluster Analyzer
        task_clusterer = WrRiverCluStream2MLPro( p_name='#1: CluStream@River',
                                                 p_n_macro_clusters = 5,
                                                 p_max_micro_clusters = 20,
                                                 p_micro_cluster_r_factor = 2,
                                                 p_time_window = 100,
                                                 p_time_gap = 10,
                                                 p_seed = 41,
                                                 p_halflife = 1.0,
                                                 p_mu = 1,
                                                 p_sigma = 1,
                                                 p_p = 2,
                                                 p_visualize=p_visualize,
                                                 p_logging=p_logging )
        
        workflow.add_task(p_task = task_clusterer)

        # 1.3 Return stream and workflow
        return stream, workflow



# 2 Prepare Demo/Unit test mode
if __name__ == '__main__':
    cycle_limit = 1000
    logging     = Log.C_LOG_ALL
    visualize   = True
    step_rate   = 2
else:
    cycle_limit = 2
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 1



# 3 Instantiate the stream scenario
myscenario = Static2DScenario(
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

clusters            = myscenario.get_workflow()._tasks[0].clusters
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