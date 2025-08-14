## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_river 
## -- Module  : howto_oa_cd_ad_007_drift_detection_ddd_1d.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-08-14  0.0.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2025-08-14)

"""

from sparccstream import *
from river.drift import DummyDriftDetector
from mlpro_int_river.wrappers.changedetectors.basics import WrDriftDetectorRiver2MLPro

from mlpro.bf import *
from mlpro.bf.events import Event
from mlpro.bf.streams.streams.clouds import StreamMLProClouds
from mlpro.oa.streams import OAStreamScenario, OAStreamWorkflow, OAStreamTask



## ------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------
class DDScenarioDDD1D (OAStreamScenario):
    
    C_NAME = 'Drift Detection with DummyDriftDetector 1D'

## ------------------------------------------------------------------------------------------------
    def _setup(self, 
               p_mode, 
               p_ada : bool, 
               p_visualize : bool, 
               p_logging,
               p_trigger_method: str = "fixed",
               p_t_0: int = 300,
               p_w: int = 0,
               p_dynamic_cloning: bool = False,
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
      
        # 3 Instantiation of River 'DummyDriftDetector' drift detector
        river_kswin = DummyDriftDetector( trigger_method = p_trigger_method,
                                          t_0 = p_t_0,
                                          w = p_w,
                                          dynamic_cloning = p_dynamic_cloning )

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
    trigger_method          = "fixed"
    t_0                     = 300
    w                       = 0
    dynamic_cloning         = False
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

    
    trigger_method          = str(input(f'Algo DDD: Thrigger method (press ENTER for {trigger_method}): ') or trigger_method)
    t_0                     = int(input(f'Algo DDD: Reference point to define triggers (press ENTER for {t_0}): ') or t_0)
    w                       = int(input(f'Algo DDD: Auxiliary parameter (press ENTER for {w}): ') or w)
    dynamic_cloning         = bool(input(f'Algo DDD: Dynamic cloning (press ENTER for {dynamic_cloning}): ') or dynamic_cloning)
    
else:
    # 1.2 Parameters for internal unit test
    cycle_limit             = 20
    logging                 = Log.C_LOG_NOTHING
    visualize               = False
    step_rate               = 1
    trigger_method          = "fixed"
    t_0                     = 300
    w                       = 0
    dynamic_cloning         = False
    drift_buffer_size       = 100
    instance_buffer_size    = 10
    detection_steprate      = 10


# 2 Instantiate the stream scenario
myscenario = DDScenarioDDD1D( p_mode = Mode.C_MODE_REAL,
                                p_cycle_limit = cycle_limit,
                                p_visualize = visualize,
                                p_logging = logging,
                                p_trigger_method = trigger_method,
                                p_t_0 = t_0,
                                p_w = w,
                                p_dynamic_cloning = dynamic_cloning,
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

