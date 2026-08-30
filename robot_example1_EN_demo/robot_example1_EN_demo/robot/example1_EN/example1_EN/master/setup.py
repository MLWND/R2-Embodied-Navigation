from setuptools import setup
import os
from glob import glob

package_name = 'master'

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
        # 安装配置文件
        (os.path.join('share', package_name, 'config'), 
         glob('config/*.yaml')),
        # 安装脚本
        (os.path.join('share', package_name, 'scripts'), 
         glob('scripts/*.sh')),
    ],
    install_requires=['setuptools', 'pyyaml'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Master controller for distributed robot control system',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'master_controller = master.master_controller:main',
            'action_manager = master.action_manager:main',
            'user_interface = master.user_interface:main',
        ],
    },
)
