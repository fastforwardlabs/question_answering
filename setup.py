from setuptools import setup, find_packages

setup(
    name='qa',
    version='0.0.3',
    description='''
        Utilities for question answering in PyTorch heavily based on 
        the 2.11.0 version of the Huggingface Transformers squad scripts.
        See https://github.com/huggingface/transformers/tree/master/examples/question-answering
    ''',
    author='Melanie Beck',
    packages=['qa']
)