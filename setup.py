from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='heliospheric_imager_analysis',
      version='0.1',
      description='A module to process and analyse Heliospheric Imager data from the STEREO-A and STEREO-B spacecraft',
      long_description=readme(),
      author='Luke Barnard',
      author_email='l.a.barnard@reading.ac.uk',
      url='https://github.com/LukeBarnard/heliospheric_imager_analysis.git',
      packages=['heliospheric_imager_analysis'],
      package_data={'heliospheric_imager_analysis': ['config.dat']},
      install_requires=['astropy', 'sunpy', 'numpy', 'scipy'],
      include_package_data=True,
      zip_safe=False)