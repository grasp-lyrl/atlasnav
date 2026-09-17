"""Python packages installed by catkin."""
from pathlib import Path
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

packages = ["core", "utils", "datasets", "datasets.gradslam_datasets"]
# Include the language feature submodule when it is initialized.
if (Path(__file__).parent / "src/clip_dinoiser/clipdino_gpu.py").is_file():
    packages.append("clip_dinoiser")
setup(**generate_distutils_setup(packages=packages, package_dir={"": "src"}))
