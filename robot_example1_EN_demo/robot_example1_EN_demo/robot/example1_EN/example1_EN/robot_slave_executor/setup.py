from setuptools import setup
import os
from glob import glob

package_name = 'robot_slave_executor'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 安装launch文件
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
        # 安装URDF文件
        (os.path.join('share', package_name, 'urdf'), 
         glob('urdf/*.urdf')),
        # 安装mesh文件
        (os.path.join('share', package_name, 'meshes'), 
         glob('meshes/*.STL')),
        # 安装RViz配置
        (os.path.join('share', package_name, 'rviz'), 
         glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ycbot',
    maintainer_email='ycbot@todo.todo',
    description='Slave executor for robot control commands',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'slave_executor_node = robot_slave_executor.slave_executor_node:main',
        ],
    },
)