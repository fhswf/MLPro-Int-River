## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_river 
## -- Module  : howto_oa_cd_ad_004_point_anomaly_detection_ocsvm_1d.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-08-12  0.0.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2025-08-12)

"""
from river.anomaly import OneClassSVM
from river import optim
from river.optim import SGD
from mlpro_int_river.wrappers.changedetectors.basics import WrAnomalyDetectorRiver2MLPro

from mlpro.bf import *
from mlpro.bf.streams.streams import StreamMLProPOutliers
from mlpro.oa.streams import OAStreamScenario, OAStreamWorkflow, OAStreamTask



## ------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------
class ADPAScenarioOCSVM1D (OAStreamScenario):
    
    C_NAME = 'Point Anomaly Detection with OneClassSVM 1D'

## ------------------------------------------------------------------------------------------------
    def _setup(self, 
               p_mode, 
               p_ada : bool, 
               p_visualize : bool, 
               p_logging,
               p_nu=0.1,
               p_optimizer: optim.base.Optimizer | None = None,
               p_intercept_lr: optim.base.Scheduler | float = 0.01,
               p_clip_gradient=1e12,
               p_initializer: optim.base.Initializer | None = None,
               p_anomaly_buffer_size: int = 100,
               p_instance_buffer_size: int = 50,
               p_detection_steprate: int = 50, 
               p_thrs_inst: int = 5,
               p_thrs_clusters: int = 1,
               **p_kwargs):
        
        # 1 Get the native stream from MLPro stream provider
        stream = StreamMLProPOutliers( p_functions = ['const'],#,'sin' , 'cos' , 'const'],
                                       p_outlier_rate=0.02,
                                       p_logging=p_logging,
                                       p_seed= 20)
        
        # 2 Creation of a workflow
        workflow = OAStreamWorkflow( p_name='wf1',
                                     p_range_max=OAStreamWorkflow.C_RANGE_NONE,
                                     p_ada=p_ada,
                                     p_visualize=p_visualize, 
                                     p_logging=p_logging )
        
        # 3 Instantiation of River 'OneClassSVM' anomaly detector
        river_ocsvm = OneClassSVM( nu = p_nu,
                                   optimizer= p_optimizer,
                                   intercept_lr= p_intercept_lr,
                                   clip_gradient= p_clip_gradient,
                                   initializer=p_initializer )
        
        # 4 Creation of tasks and add them to the workflow
        anomalydetector = WrAnomalyDetectorRiver2MLPro( p_algo_river = river_ocsvm,
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
    nu                      = 0.1
    optimizer               = None
    intercept_lr            = 0.01
    clip_gradient           = 1e12
    initializer             = None
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

    nu                      = (input(f'Algo OCSVM: Nu (press ENTER for {nu}): ') or nu)
    optimizer               = optim.base.Optimizer(input(f'Algo OCSVM: The sequencial optimizer (press ENTER for {optimizer}): ') or optimizer)
    intercept_lr            = optim.base.Optimizer(input(f'Algo OCSVM: Learning rate scheduler (press ENTER for {intercept_lr}): ') or intercept_lr)
    clip_gradient           = (input(f'Algo OCSVM: Clip gradient (press ENTER for {clip_gradient}): ') or clip_gradient)
    initializer             = optim.base.Optimizer(input(f'Algo OCSVM: Weights initialization scheme (press ENTER for {initializer}): ') or initializer)

else:
    # 1.2 Parameters for internal unit test
    cycle_limit             = 20
    logging                 = Log.C_LOG_NOTHING
    visualize               = False
    step_rate               = 1
    nu                      = 0.1
    optimizer               = None
    intercept_lr            = 0.01
    clip_gradient           = 1e12
    initializer             = None
    anomaly_buffer_size     = 100
    instance_buffer_size    = 10
    detection_steprate      = 10


# 2 Instantiate the stream scenario
myscenario = ADPAScenarioOCSVM1D( p_mode = Mode.C_MODE_REAL,
                                  p_cycle_limit = cycle_limit,
                                  p_visualize = visualize,
                                  p_logging = logging,
                                  p_nu = nu,
                                  p_optimizer = optimizer,
                                  p_intercept_lr = intercept_lr,
                                  p_clip_gradient = clip_gradient,
                                  p_initializer = initializer,                          
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