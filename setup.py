from setuptools import setup, find_packages

setup(
    name='wml',
    version='3.0.0',
    packages=find_packages(),
    install_requires=[
        # 列出你的依赖包，例如：
        # 'requests>=2.20.0',
    ],
    author='WangJie',
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

