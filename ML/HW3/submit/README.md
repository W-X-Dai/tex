# ML HW3 Code

This directory contains the implementation for Homework 3 of the Machine Learning course.
It includes four main files:

- `ann.py`: Artificial Neural Network (ANN) for binary classification.
- `ldf.py`: Linear Discriminant Function (LDF) for binary classification.
- `svm.py`: Support Vector Machine (SVM) implemented from scratch.
- `svm_sklearn.py`: Support Vector Machine (SVM) using scikit-learn.

## Requirements

- Ubuntu 22.04 / 24.04 (or WSL)
- Python >= 3.10
- g++ >= 11
- pybind11
- Eigen3
- numpy, scikit-learn

## Compilation Instructions
To execute `ann.py` and `ldf.py`, please compile `myann_module.cpp` first with the following command.
```
c++ -O3 -Wall -shared -std=c++17 -fPIC \
  $(python3 -m pybind11 --includes) \
  -I/usr/include/eigen3 \
  ./myann_module.cpp \
  -o myann$(python3-config --extension-suffix)
```

If your os havn't installed `eigen3`, you can install it via conda:
```
sudo apt update
sudo apt install libeigen3-dev
```

Finally, please install `pybind11` via pip if you haven't installed it yet:

```
pip install pybind11
```

Note:
You may need to install a newer C++ runtime via:

```
conda install -c conda-forge libstdcxx-ng
```
if you encounter errors related to C++14 or C++17 features during compilation or execution.