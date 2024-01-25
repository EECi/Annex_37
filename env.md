# Setting up Python environment

Due to the large number of package dependencies, there are conflicts in the package requirements across the top-level dependencies. Combined with the need to use a CUDA compatible version of PyTorch, the following steps are required to set up the environment.

1. Create a new virtual environment with Python 3.8

2. ```pip install -r requirements.txt``` - install top-level dependencies

3. ```pip3 install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html``` - install CUDA compatible PyTorch

4. ```pip uninstall torchvision``` - remove `torchvision` which causes irrelevant warnings