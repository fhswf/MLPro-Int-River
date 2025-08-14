## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_river 
## -- Module  : howto_oa_cd_ad_010_drift_detection_ph_1d.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-08-14  0.0.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2025-08-14)

"""

from sparccstream import *
from river.drift import PageHinkley
from mlpro_int_river.wrappers.changedetectors.basics import WrDriftDetectorRiver2MLPro

from mlpro.bf import *
from mlpro.bf.events import Event
from mlpro.bf.streams.streams.clouds import StreamMLProClouds
from mlpro.oa.streams import OAStreamScenario, OAStreamWorkflow, OAStreamTask



## ------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------
class DDScenarioPH1D (OAStreamScenario):
    
    C_NAME = 'Drift Detection with PageHinkley 1D'

## ------------------------------------------------------------------------------------------------
    def _setup(self, 
               p_mode, 
               p_ada : bool, 
               p_visualize : bool, 
               p_logging,
               p_min_instances:int = 30,
               p_delta: float = 0.005,
               p_threshold: float = 50.0,
               p_alpha: float = 0.9999,
               *,
               p_r_algo_mode: str = "both",
               p_drift_buffer_size: int = 100,
               p_instance_buffer_size: int = 20,
               p_detection_steprate:int = 1,
               **p_kwargs):
        
        # 1 Get the native stream from MLPro stream provider
        stream = StreamMLProClouds( p_num_dim = 1,
                                    p_num_instances= 50,
                                    p_num_clouds= 2,
                                    p_radii= [50],
                                    p_weights=[1,2],
                                    p_seed= 20)
        
        # 2 Creation of a workflow
        workflow = OAStreamWorkflow( p_name='wf1',
                                     p_range_max=OAStreamWorkflow.C_RANGE_NONE,
                                     p_ada=p_ada,
                                     p_visualize=p_visualize, 
                                     p_logging=p_logging )
      
        # 3 Instantiation of River 'PageHinkley' drift detector
        river_kswin = PageHinkley( min_instances = p_min_instances,
                                   delta = p_delta,
                                   threshold = p_threshold,
                                   alpha = p_alpha,
                                   mode = p_r_algo_mode )

        # 4 Creation of tasks and add them to the workflow
        driftdetector = WrDriftDetectorRiver2MLPro( p_algo_river = river_kswin,
                                                    p_drift_buffer_size = p_drift_buffer_size,
                                                    p_instance_buffer_size = p_instance_buffer_size,
                                                    p_detection_steprate = p_detection_steprate,
                                                    p_visualize = p_visualize,
                                                    p_logging = p_logging )
        
        workflow.add_task(p_task = driftdetector )

        # 5 Return stream and workflow
        return stream, workflow



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
# 1 Preparation of demo/unit test mode
if __name__ == "__main__":
    # 1.1 Parameters for demo mode
    cycle_limit             = 500
    logging                 = Log.C_LOG_WE
    step_rate               = 1
    min_instances           = 30
    delta                   = 0.05
    threshold               = 50.0
    alpha                   = 0.9999
    r_algo_mode             = "both"
    drift_buffer_size       = 100
    instance_buffer_size    = 50
    detection_steprate      = 50

    cycle_limit             = int(input(f'\nCycle limit (press ENTER for {cycle_limit}): ') or cycle_limit)
    visualize               = input('Visualization Y/N (press ENTER for Y): ').upper() != 'N'
    if visualize:
        i = input(f'Visualization step rate (press ENTER for {step_rate}): ')
        if i != '': step_rate = int(i)

        i = input('Log level: "A"=All, "W"=Warnings only, "N"=Nothing (press ENTER for "W"): ').upper() 
        if i == 'A': logging = Log.C_LOG_WE
        elif i == 'N': logging = Log.C_LOG_NOTHING

    
    min_instances           = int(input(f'Algo PH: Minimum number of instances (press ENTER for {min_instances}): ') or min_instances)
    delta                   = float(input(f'Algo PH: Delta (press ENTER for {delta}): ') or delta)
    threshold               = float(input(f'Algo PH: Change detection threshold (press ENTER for {threshold}): ') or threshold)
    alpha                   = float(input(f'Algo PH: Alpha (press ENTER for {alpha}): ') or alpha)
    r_algo_mode             = str(input(f'Algo PH: Mode (press ENTER for {r_algo_mode}): ') or r_algo_mode)

else:
    # 1.2 Parameters for internal unit test
    cycle_limit             = 20
    logging                 = Log.C_LOG_NOTHING
    visualize               = False
    step_rate               = 1
    min_instances           = 30
    delta                   = 0.05
    threshold               = 50.0
    alpha                   = 0.9999
    r_algo_mode             = "both"
    drift_buffer_size       = 100
    instance_buffer_size    = 10
    detection_steprate      = 10


# 2 Instantiate the stream scenario
myscenario = DDScenarioPH1D( p_mode = Mode.C_MODE_REAL,
                                p_cycle_limit = cycle_limit,
                                p_visualize = visualize,
                                p_logging = logging,
                                p_min_instances = min_instances,
                                p_delta = delta,
                                p_threshold = threshold,
                                p_alpha = alpha,
                                p_r_algo_mode = r_algo_mode,
                                p_drift_buffer_size = drift_buffer_size,
                                p_instance_buffer_size = instance_buffer_size,
                                p_detection_steprate = detection_steprate )

if visualize:
    myscenario.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                        p_view_autoselect = True,
                                                        p_step_rate = step_rate ) )


# 3 Reset and run own stream scenario
myscenario.reset()

if __name__ == '__main__':
    input('Press ENTER to start stream processing...')

myscenario.run()


if __name__ == '__main__':
    input('Press ENTER to exit...')

