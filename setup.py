import setuptools
import pathlib

def parse_requirements(filename):
    with open(filename, 'r') as file:
        lines = (line.strip() for line in file)
        return [line for line in lines if line and not line.startswith('#')]

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
        'gym[atari]', 'atari_py', 'crafter', 'dm_control', 'ruamel.yaml',
        'tensorflow', 'tensorflow_probability'], install_requires=parse_requirements('requirements.txt'),
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
