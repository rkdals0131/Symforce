from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'cc_slam_sym'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'docs'), glob('docs/*.md')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user1',
    maintainer_email='kikiws70@gmail.com',
    description='Cone Cluster SLAM using GTSAM and Symforce for Formula Student Driverless',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'dummy_publisher = cc_slam_sym.dummy_publisher_node:main',
        ],
    },
)
