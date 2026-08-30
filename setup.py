from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'k9_perception_pkg'

setup(
    name=package_name,
    version='0.0.1',

    packages=find_packages(
        exclude=['test']
    ),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join(
                'share',
                package_name,
                'config'
            ),
            glob('config/*.yaml')
        ),
        (
            os.path.join(
                'share',
                package_name,
                'models'
            ),
            glob('models/*.onnx')
        ),
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='hopkira',
    maintainer_email='hopkira@todo.todo',

    description='K9 perception nodes',

    license='Apache-2.0',

    entry_points={
        'console_scripts': [
            'face_detector = '
            'k9_perception_pkg.face_detector_node:main',
        ],
    },
)
