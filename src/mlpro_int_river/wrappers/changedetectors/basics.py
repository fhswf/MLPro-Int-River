## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_river.wrappers.changedetectors
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-08-12  0.0.0     DS       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2025-08-12)

This module provides wrapper root classes from River to MLPro, specifically for anomaly detectors. 

Learn more:
https://www.riverml.xyz/

"""

import numpy as np
from river.anomaly.base import AnomalyDetector as ADRiver
from river.base import DriftDetector as DDRiver

from mlpro.bf import Log, ParamError
from mlpro.bf.streams import StreamTask, Instance
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.instancebased import AnomalyDetectorIBPG
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.instancebased import DriftDetectorIB
from mlpro.oa.streams.tasks.changedetectors.anomalydetectors.anomalies.instancebased import PointAnomaly
from mlpro.oa.streams.tasks.changedetectors.driftdetectors.clusterbased.generic import DriftDetectorCBGenSingleGradient
from mlpro_int_river.wrappers import WrapperRiver



# Export list for public API
__all__ = [ 'WrAnomalyDetectorRiver2MLPro',
            'WrDriftDetectorRiver2MLPro' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrAnomalyDetectorRiver2MLPro (AnomalyDetectorIBPG, WrapperRiver):

    C_TYPE = 'Anomaly Detector (river)'

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_algo_river : ADRiver, 
                 p_range_max=StreamTask.C_RANGE_THREAD, 
                 p_ada = True, p_duplicate_data = False, 
                 p_visualize = False, 
                 p_logging=Log.C_LOG_ALL, 
                 p_anomaly_buffer_size = 100, 
                 p_thrs_inst = 0, 
                 p_group_anomaly_det = True, 
                 p_instance_buffer_size : int = 20,
                 p_detection_steprate : int = 1,
                 **p_kwargs):
        
        WrapperRiver.__init__( self, p_logging = p_logging )

        AnomalyDetectorIBPG.__init__( self, 
                                      p_group_anomaly_det = p_group_anomaly_det, 
                                      p_name = type(p_algo_river).__name__, 
                                      p_range_max = p_range_max, 
                                      p_ada = True, 
                                      p_duplicate_data = p_duplicate_data, 
                                      p_visualize = p_visualize, 
                                      p_logging = p_logging, 
                                      p_anomaly_buffer_size = p_anomaly_buffer_size,
                                      p_thrs_inst = 1,
                                      **p_kwargs )
        
        if ( p_detection_steprate > p_instance_buffer_size ) or ( p_detection_steprate < 1 ):
            raise ParamError('Please set the parameter "p_detection_steprate" >= 1 and <= "p_instance_buffer_size"')
        
        self._algo_river                = p_algo_river
        self._inst_buffer_size          = p_instance_buffer_size
        self._detection_steprate        = p_detection_steprate
        self._inst_counter              = 0

        self._inst_data_buffer : np.ndarray  = None
        self._inst_data_buffer_full : bool   = False
        self._inst_ref_buffer : np.ndarray   = np.empty(self._inst_buffer_size, dtype = object)

        self._inst_buffer_pos : int          = 0

        self._block_mode = ( self._detection_steprate == self._inst_buffer_size )
        

## -------------------------------------------------------------------------------------------------
    def _detect(self, p_instance : Instance, **p_kwargs):
 
               
        feature_data   = p_instance.get_feature_data()
        feature_values = feature_data.get_values()
        num_features   = feature_data.get_related_set().get_num_dim()
        feature_names  = feature_data.get_dim_ids()

        instance_dict = dict(zip(feature_names, feature_values))
        score = self._algo_river.score_one(instance_dict)

        if score > 0.8: 
            anomaly = PointAnomaly( p_status=True,
                                    p_tstamp=p_instance.tstamp,
                                    p_visualize=self.get_visualization(),
                                    p_raising_object=self,
                                    p_instances=[p_instance])
            
            self._raise_anomaly_event(p_anomaly=anomaly, p_instance=p_instance)

        self._algo_river.learn_one(instance_dict)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrDriftDetectorRiver2MLPro (DriftDetectorIB, WrapperRiver):

    C_TYPE = 'Drift Detector (river)'

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_algo_river : DDRiver, 
                 p_range_max=StreamTask.C_RANGE_THREAD, 
                 p_ada = True, 
                 p_duplicate_data = False, 
                 p_visualize = False, 
                 p_logging=Log.C_LOG_ALL, 
                 p_drift_buffer_size = 100, 
                 p_thrs_inst = 0, 
                 p_instance_buffer_size : int = 20,
                 p_detection_steprate : int = 1,
                 p_feature_idx=0,
                 p_feature_id=None,
                 **p_kwargs):
        
        WrapperRiver.__init__( self, p_logging = p_logging )

        DriftDetectorIB.__init__( self, 
                                  p_name = type(p_algo_river).__name__, 
                                  p_range_max = p_range_max, 
                                  p_ada = True, 
                                  p_duplicate_data = p_duplicate_data, 
                                  p_visualize = p_visualize, 
                                  p_logging = p_logging, 
                                  p_drift_buffer_size = p_drift_buffer_size,
                                  p_thrs_inst = 1,
                                  **p_kwargs )
        
        if ( p_detection_steprate > p_instance_buffer_size ) or ( p_detection_steprate < 1 ):
            raise ParamError('Please set the parameter "p_detection_steprate" >= 1 and <= "p_instance_buffer_size"')
        
        self._algo_river                = p_algo_river
        self._feature_idx               = p_feature_idx
        self._feature_id                = p_feature_id
        self._inst_buffer_size          = p_instance_buffer_size
        self._detection_steprate        = p_detection_steprate
        self._inst_counter              = 0

        self._inst_data_buffer : np.ndarray  = None
        self._inst_data_buffer_full : bool   = False
        self._inst_ref_buffer : np.ndarray   = np.empty(self._inst_buffer_size, dtype = object)

        self._inst_buffer_pos : int          = 0

        self._block_mode = ( self._detection_steprate == self._inst_buffer_size )
        

## -------------------------------------------------------------------------------------------------
    def _detect(self, p_instance : Instance, **p_kwargs):

        # Get input feature vector
        feature_data = p_instance.get_feature_data()

        # Determine feature value to monitor
        if self._feature_id is not None:
            value = feature_data.get_value(self._feature_id)
        else:
            value = feature_data.get_values()[self._feature_idx]

        # Feed value into River drift detector
        self._algo_river.update(value)

        if self._algo_river.drift_detected:
            drift = DriftDetectorCBGenSingleGradient(
                p_status=True,
                p_tstamp=p_instance.tstamp,
                p_visualize=self.get_visualization(),
                p_raising_object=self,
                p_instances=[p_instance]
            )

            self._raise_drift_event(p_drift=drift, p_instance=p_instance)

