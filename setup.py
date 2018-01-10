from setuptools import setup
import argiope

setup(name='argiope',
      version=argiope.__version__,
      description="A framework for simpler finite element processing",
      long_description="",
      author='Ludovic Charleux, Emile Roux',
      author_email='ludovic.charleux@univ-smb.fr',
      license='GPL v3',
      packages=['argiope'],
      zip_safe=False,
      url='https://github.com/lcharleux/argiope',
      install_requires=[
          "numpy",
          "scipy",
          "matplotlib",
          "pandas",
          "jupyter",
          "nbconvert"
          ],
      )
