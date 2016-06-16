from setuptools import setup

setup(name='argiope',
      version='0.0',
      description="A framework for simpler finite element processing",
      long_description="",
      author='Ludovic Charleux',
      author_email='ludovic.charleux@univ-smb.fr',
      license='GPL v3',
      packages=['argiope'],
      zip_safe=False,
      install_requires=[
          "numpy",
          "scipy",
          "matplotlib",
          "pandas"
          ],
      )
