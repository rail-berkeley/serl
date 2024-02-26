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
        "agentlace@git+https://github.com/youliangtan/agentlace.git@e61032fbce8a1e6d3dc2aeba21de082e4bf46fe3",
    ],
    packages=find_packages(),
    zip_safe=False,
)
