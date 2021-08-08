from setuptools import setup, find_packages

setup(name="PhyTimeSimulator",
      version='0.0.0',
      keywords='World, Solid, Liquid, Planet, Field, Observe',
      description='A physics engine',
      license='MIT License',
      packages=find_packages,
      include_package_data=True,
      platforms='any',
      install_requires=['numpy'],
      zip_safe=False

      )
