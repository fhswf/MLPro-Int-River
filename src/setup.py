from setuptools import setup


setup(name='mlpro_int_river',
version='1.0.0',
description='MLPro: Integration River',
author='MLPro Team',
author_mail='mlpro@listen.fh-swf.de',
license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
packages=['mlpro_int_river'],

# Package dependencies for full installation
extras_require={
    "full": [
        "dill==0.3.6",
        "numpy==1.23.5",
        "matplotlib==3.6.3",
        "transformations==2022.9.26",
        "scipy==1.8.1",
        "pymunk==6.4.0",
        "multiprocess==0.70.14",
        "scikit-learn==1.2.0",
        "optuna==3.0.5",
        "hyperopt==0.2.7",
        "pyglet==1.5.27",
        "mlpro>=2.0.0"
        "river==0.14.0"
    ],
},

zip_safe=False)