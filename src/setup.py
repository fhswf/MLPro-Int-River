from setuptools import setup


setup(name='mlpro-int-river',
version='0.5.0',
description='MLPro: Integration River',
author='MLPro Team',
author_mail='mlpro@listen.fh-swf.de',
license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
packages=['mlpro_int_river'],

# Package dependencies for full installation
extras_require={
    "full": [
        "mlpro[full]>=2.1.0",
        "river>=0.22.0"
    ],
},

zip_safe=False)