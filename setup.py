from setuptools import setup, find_packages

setup(
    name='selective-search-pytorch',
    version='0.1',
    description='Fork of selective_search_pytorch with scriptable API, smoothness-based mask selection, and bounding box extraction',
    author='Your Name',
    author_email='your@email.com',
    url='https://github.com/yourusername/selective_search_pytorch',
    # packages=find_packages(include=['selective_search', 'selective_search.*', 'opencv_custom', 'opencv_custom.*']),
    packages=find_packages(),
    # package_data={
    #     'selective_search': ['../opencv_custom/selectivesearchsegmentation_opencv_custom_.so']
    # },
    package_data={
        'opencv_custom': [
            'selectivesearchsegmentation_opencv_custom.py',
            'selectivesearchsegmentation_opencv_custom_.so'
        ]
    },
    include_package_data=True,
    install_requires=[
        'torch',
        'opencv-python',
        'numpy',
        'Pillow',
        'matplotlib',
        'torchvision',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)