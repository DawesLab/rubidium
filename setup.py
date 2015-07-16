from setuptools import setup

setup(name='rubidium',
      version='1.0',
      description='Model Rubidium D1 and D2 spectra',
      url='http://github.com/DawesLab/rubidium',
      author='Andrew M.C. Dawes',
      author_email='dawes@pacificu.edu',
      license='GPLv3',
      packages=['rubidium'],
      install_requires=[
          'numpy','scipy','matplotlib'
      ],
      zip_safe=False)
