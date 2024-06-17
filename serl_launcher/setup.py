from setuptools import setup, find_packages

setup(
    name="serl_launcher",
    version="0.1.3",
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
        "agentlace@git+https://github.com/youliangtan/agentlace.git@f025024631db0992a90085ee4637d8c0c90da317",
    ],
    packages=find_packages(),
    zip_safe=False,
)
