from setuptools import setup
setup(name='franka_env',
      version='0.0.1',
      install_requires=['gym',
                        'pypylon',
                        'pyrealsense2',
                        'opencv-python',
                        'pyquaternion',
                        'hidapi',
                        'pyyaml',
                        'rospkg',
                        'scipy',
                        'requests',
                        'Pillow',
                        'flask',
                        'defusedxml']
)