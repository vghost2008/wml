from setuptools import setup, find_packages

setup(
    name='python-wml',
    version='3.0.1',
    packages=find_packages(),
    include_package_data=True,
    include_dirs=True,
    install_requires=[
    'opencv-python>=4.5.3.56',
    'pycocotools>=2.0.7',
    'gast>=0.2.2',
    'yacs>=0.1.8',
    'scikit-learn',
    'pandas>=0.23.0',
    'PyTurboJPEG',
    'einops',
    'easydict',
    ],
    author='vghost2008',
    author_email='bluetornado@zju.edu.cn',
    description='Deep Learning toolkit',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/vghost2008/wml',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

