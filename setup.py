import pathlib
from setuptools import setup

dir_here = pathlib.Path(__file__).parent
README = (dir_here / 'README.md').read_text()

setup(
    name='trialML',
    version='0.1.0',    
    description='trialML package',
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://github.com/ErikinBC/trialML',
    author='Erik Drysdale',
    author_email='erikinwest@gmail.com',
    license='MIT',
    packages=['trialML'],
    package_data={'trialML': ['_datagen/*']},
    include_package_data=True,
    install_requires=['numpy', 'pandas'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
    ],
)

