from setuptools import setup, find_packages

setup(name="SimpleGeo",
      version='0.0.0',
      description='Geometry',
      license='MIT License',
      packages=find_packages,
      include_package_data=True,
      platforms='any',
      install_requires=['numpy', 'sympy'],
      zip_safe=False

      )
