## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_river 
## -- Module  : howto_oa_cd_ad_015_point_anomaly_detection_pad_3d.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-08-12  0.0.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2025-08-12)

"""
from river.anomaly import PredictiveAnomalyDetection
from river import optim,base
from mlpro_int_river.wrappers.changedetectors.basics import WrAnomalyDetectorRiver2MLPro

from mlpro.bf import *
from mlpro.bf.streams.streams import StreamMLProPOutliers
from mlpro.oa.streams import OAStreamScenario, OAStreamWorkflow, OAStreamTask



## ------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------
class ADPAScenarioPAD3D (OAStreamScenario):
    
    C_NAME = 'Point Anomaly Detection with PredictiveAnomalyDetection 3D'

## ------------------------------------------------------------------------------------------------
    def _setup(self, 
               p_mode, 
               p_ada : bool, 
               p_visualize : bool, 
               p_logging,
               p_predictive_model: base.Estimator|None = None,
               p_horizon: int = 1,
               p_n_std: float = 3.0,
               p_warmup_period: int = 0,
               p_anomaly_buffer_size: int = 100,
               p_instance_buffer_size: int = 50,
               p_detection_steprate: int = 50, 
               **p_kwargs):
        
        # 1 Get the native stream from MLPro stream provider
        stream = StreamMLProPOutliers( p_functions = ['const', 'const', 'const'],#,'sin' , 'cos' , 'const'],
                                       p_outlier_rate=0.02,
                                       p_logging=p_logging,
                                       p_seed= 20)
        
        # 2 Creation of a workflow
        workflow = OAStreamWorkflow( p_name='wf1',
                                     p_range_max=OAStreamWorkflow.C_RANGE_NONE,
                                     p_ada=p_ada,
                                     p_visualize=p_visualize, 
                                     p_logging=p_logging )
        
        # 3 Instantiation of River 'LocalOutlierFactor' anomaly detector
        river_pad = PredictiveAnomalyDetection( predictive_model = p_predictive_model,
                                                horizon  = p_horizon,
                                                n_std = p_n_std,
                                                warmup_period = p_warmup_period)
        
        # 4 Creation of tasks and add them to the workflow
        anomalydetector = WrAnomalyDetectorRiver2MLPro( p_algo_river = river_pad,
                                                        p_anomaly_buffer_size = p_anomaly_buffer_size,
                                                        p_instance_buffer_size = p_instance_buffer_size,
                                                        p_detection_steprate = p_detection_steprate,
                                                        p_group_anomaly_det = False,
                                                        p_visualize = p_visualize,
                                                        p_logging = p_logging )
        
        workflow.add_task(p_task = anomalydetector)
        
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
    predictive_model        = None
    horizon                 = 1
    n_std                   = 3.0
    warmup_period           = 0
    anomaly_buffer_size     = 100
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

    predictive_model        = (input(f'Algo PAD: Predictive model (press ENTER for {predictive_model}): ') or predictive_model)
    horizon                 = int(input(f'Algo PAD: Horizon (press ENTER for {horizon}): ') or horizon)
    n_std                   = float(input(f'Algo PAD: Number of Standard Deviations (press ENTER for {n_std}): ') or n_std)
    warmup_period           = int(input(f'Algo PAD: Warmup period (press ENTER for {warmup_period}): ') or warmup_period)

else:
    # 1.2 Parameters for internal unit test
    cycle_limit             = 20
    logging                 = Log.C_LOG_NOTHING
    visualize               = False
    step_rate               = 1
    n_neighbors             = 10
    distance_func           = None
    anomaly_buffer_size     = 100
    instance_buffer_size    = 10
    detection_steprate      = 10


# 2 Instantiate the stream scenario
myscenario = ADPAScenarioPAD3D( p_mode = Mode.C_MODE_REAL,
                                p_cycle_limit = cycle_limit,
                                p_visualize = visualize,
                                p_logging = logging,
                                p_predictive_model = predictive_model,
                                p_horizon = horizon,      
                                p_n_std = n_std,
                                p_warmup_period = warmup_period,                
                                p_anomaly_buffer_size = anomaly_buffer_size,
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