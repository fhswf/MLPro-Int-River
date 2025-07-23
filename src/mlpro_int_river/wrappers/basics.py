## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro_int_river.wrappers
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-06-14  0.0.0     LSB      Creation
## -- 2022-06-14  1.0.0     LSB      Release of first version
## -- 2022-06-18  1.0.1     LSB      Stream names as Stream ids
## -- 2022-06-23  1.0.2     LSB      Meta data and instances in Numpy format
## -- 2022-06-25  1.0.3     LSB      Refactoring for label and instance class
## -- 2022-08-15  1.1.0     DA       Introduction of root class Wrapper
## -- 2022-11-03  1.1.1     LSB      Bug fix for river update
## -- 2022-11-03  1.2.0     DA       - Refactoring
## --                                - Class WrStreamRiver: removed parent class Wrapper
## -- 2022-11-07  1.3.0     DA       Class WrStreamOpenML: refactoring to make it iterable
## -- 2022-11-08  1.3.1     DA       Corrections
## -- 2022-11-19  1.4.0     DA       Method WrStreamRiver._get_string(): new parameter p_name
## -- 2022-12-09  1.4.1     DA       Bugfix
## -- 2023-05-12  2.0.0     DA       - New root class WrapperRiver
## --                                - New wrappers for River cluster analyzers
## --                                - Refatoring of classes WrStream*
## --                                - Class WrStreamProviderRiver: detects now all River data sets
## --                                - Conversion to subpackage with modules for
## --                                    - native River streams
## --                                    - selected cluster algorithms
## --                                    - selected algorithms for anomaly detection 
## -- 2024-02-15  2.1.0     DA       Updated minimum version of river to 0.21.0
## -- 2024-04-10  2.1.1     DA       Refactoring
## -- 2025-07-21  2.1.2     DS       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.1.2 (2025-07-21)

This module contains the abstract root class for all river wrapper classes.

Learn more:
https://www.riverml.xyz/

"""

from mlpro.bf.various import ScientificObject
from mlpro.wrappers import Wrapper

# Export list for public API
__all__ = [ 'WrapperRiver' ]



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrapperRiver (Wrapper):
    """
    Root class for all River wrapper classes.
    """

    C_TYPE              = 'Wrapper River'
    C_WRAPPED_PACKAGE   = 'river'
    C_MINIMUM_VERSION   = '0.21.0'

    C_SCIREF_TYPE       = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR     = 'River'
    C_SCIREF_URL        = 'riverml.xyz'
