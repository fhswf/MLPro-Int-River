from setuptools import setup


setup(name='mlpro_int_river',
version='0.1.0',
description='MLPro: Integration River',
author='MLPro Team',
author_mail='mlpro@listen.fh-swf.de',
license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
packages=['mlpro_int_river'],

# Package dependencies for full installation
extras_require={
    "full": [
        "dill",
        "numpy",
        "matplotlib",
        "multiprocess",
        "mlpro>=1.3.1",
        "river==0.15.0"
    ],
},

zip_safe=False)