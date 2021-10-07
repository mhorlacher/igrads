from setuptools import setup, find_packages

requirements = [
    "pandas==1.2.4",
    "numpy==1.19.4",
    "tensorflow>=2.0.0",
    "logomaker"
]

setup(name='igrads',
      version='0.1',
      description='Integrated Gradients for Tensorflow 2.x',
      url='http://github.com/mhorlacher/deeple',
      author='Marc Horlacher',
      author_email='marc.horlacher@helmholtz-muenchen.de',
      license='MIT',
      install_requires=requirements,
      include_package_data=True,
      packages=find_packages(),
      zip_safe=False)