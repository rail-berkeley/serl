from setuptools import setup, find_packages

setup(
    name="serl_launcher",
    version="0.1.2",
    description="library to enable distributed edge ml training and inference",
    url="https://github.com/youliangtan/edgeml",
    author="auth",
    author_email="tan_you_liang@hotmail.com",
    license="MIT",
    install_requires=[
        "zmq",
        "typing",
        "typing_extensions",
        "opencv-python",
        "lz4",
        "edgeml@git+https://github.com/youliangtan/edgeml.git@e52618a",
    ],
    packages=find_packages(),
    zip_safe=False,
)
