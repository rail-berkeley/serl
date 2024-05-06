from setuptools import setup, find_packages

setup(
    name="serl_launcher",
    version="0.1.2",
    description="library for rl experiments",
    url="https://github.com/rail-berkeley/serl",
    author="auth",
    license="MIT",
    install_requires=[
        "zmq",
        "typing",
        "typing_extensions",
        "opencv-python",
        "lz4",
        "agentlace@git+https://github.com/youliangtan/agentlace.git@892d1557264d7bb1d5df04b37638c850c9d36f35",
    ],
    packages=find_packages(),
    zip_safe=False,
)
