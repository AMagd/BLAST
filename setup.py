import setuptools
import pathlib


setuptools.setup(
    name='dreamerv2',
    version='2.2.0',
    description='Mastering Atari with Discrete World Models',
    url='http://github.com/danijar/dreamerv2',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=['dreamerv2', 'dreamerv2.common'],
    package_data={'dreamerv2': ['configs.yaml']},
    entry_points={'console_scripts': ['dreamerv2=dreamerv2.train:main']},
    install_requires=[
        'tensorflow==2.6.0', 'tensorflow_probability==0.14.1', 'ruamel.yaml', 'gym[atari]', 'dm_control', 'pudb', 'gym-minigrid==1.0.2', 'sk-video', 
        'gym-minigrid==1.0.2', 'gym-notices==0.0.8', 'gymnasium==0.27.1', 'gymnasium-notices==0.0.1', 'numpy==1.23.5', 'matplotlib', 'protobuf==3.19.6', 'keras==2.6.0'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)

