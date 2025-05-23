.. MLPro Documentations documentation master file, created by
   sphinx-quickstart on Wed Sep 15 12:06:53 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MLPro-Int-River - Integration of River into MLPro
=================================================

Welcome to MLPro-Int-River, an extension to MLPro to integrate the River package.
MLPro is a middleware framework for standardized machine learning in Python. It is 
developed by the South Westphalia University of Applied Sciences, Germany, and provides 
standards, templates, and processes for hybrid machine learning applications. River, in 
turn, provides numerous state-of-the-art algorithms for several topics of online machine 
learning.

MLPro-Int-River provides wrapper classes that enable the use of selected River functionalities 
in your MLPro applications. The use of these wrappers is illustrated in numerous example programs.

.. image:: MLPro-Int-River.gif
   :width: 300


**Preparation**

Before running the examples, please install the latest versions of MLPro, River, and MLPro-Int-River as follows:

.. code-block:: bash

   pip install mlpro-int-river[full] --upgrade


**See also**
   - `MLPro - The integrative middleware framework for standardized machine learning in Python <https://mlpro.readthedocs.io>`_ 
   - `MLPro-OA - Sub-framework for online machine learning <https://mlpro.readthedocs.io/en/latest/content/03_machine_learning/mlpro_oa/main.html>`_
   - `River - Online machine learning in Python <https://riverml.xyz>`_ 
   - `MLPro-Int-River on GitHub <https://github.com/fhswf/MLPro-Int-River>`_
   - `MLPro-Int-River on PyPI <https://pypi.org/project/mlpro-int-river>`_
   - `Further MLPro extensions <https://mlpro.readthedocs.io/en/latest/content/04_extensions/main.html>`_


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Home

   self


.. toctree::
   :maxdepth: 2
   :caption: Example Pool
   :glob:

   content/01_example_pool/*


.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :glob:

   content/02_api/*


.. toctree::
   :maxdepth: 2
   :caption: About
   :glob:

   content/03_about/*
