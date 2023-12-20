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
        "edgeml@git+https://github.com/youliangtan/edgeml.git@833332d67dc42617b4479e9b87d8192f36296cb9",
    ],
    packages=find_packages(),
    zip_safe=False,
)
