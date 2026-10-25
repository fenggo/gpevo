

# USPEX-GP

## Introduction

USPEX-GP is an open-source computational materials science package designed for crystal structure prediction using  Gaussian process guided evolutionary algorithms (based on USPEX-9.4.4). It supports various computational codes (VASP, Quantum ESPRESSO, LAMMPS, etc.) and is designed for high-performance computing environments. This repository contains a hybrid codebase with both MATLAB main scripts and Python helper tools/extensions.

## Features

* **Crystal Structure Prediction**: Predicts stable crystal structures given only chemical composition.
* **Evolutionary Algorithm (EA)**: Uses a robust genetic algorithm with local optimization.
* **Particle Swarm Optimization (PSO)**: Alternative optimization algorithm included.
* **Variable Composition Search**: Supports searching across different chemical compositions.
* **Molecule & Surface Predictions**: Specialized modes for molecular crystals and surface structures.
* **Metadynamics**: Enhanced sampling capabilities for complex energy landscapes.
* **Multi-Code Support**: Interfaces with popular DFT and MD codes.
* **Hybrid MATLAB/Python Architecture**: Combines OCTAVE's numerical power with Python's flexibility.

## Installation

### Prerequisites

* **OCTAVE** with toolboxes:
  * Optimization Toolbox
  * Statistics and Machine Learning Toolbox
* **Python** (3.7 or newer)
* **Cython** (for compiling core extensions)
* **Numerical Python (NumPy)**
* **C Compiler** (e.g., GCC, MinGW) compatible with your Python/OCTAVE setup.

### Steps

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd uspex-gp
   ```

2. **Setup Python Environment**:
   It is recommended to use a virtual environment.
   ```bash
   pip install numpy cython
   ```

3. **Compile Cython Extensions**:
   Navigate to the `FunctionFolder/sys` directory and compile the extensions (refer to specific `.pyx` files or local build instructions if provided).
   ```bash
   python setup.py build_ext --inplace
   ```

4. **Python package** (for the `gpevo` command):
   ```bash
   cd ~/gpevo
   /home/feng/.local/anaconda/bin/python -m pip install .
   ```
   This installs the `gpevo` console command and bundles the Python
   implementation, space-group data, and scheduler clients used for cluster
   submissions. The wheel also compiles the `random_cell_mol_cy` Cython
   extension; the code falls back to the pure-Python `random_cell_mol.py` if
   the compiled extension is unavailable. The run directory only needs
   `INPUT.txt` and the `MOL_*` inputs; no `USPEXPATH` environment variable is
   required. The legacy MATLAB source tree remains available but is not needed
   by the Python command.

5. **MATLAB Configuration** (legacy path):
   Add the `FunctionFolder` and its subdirectories to your MATLAB path. You can do this via the MATLAB interface or by adding a path file in your startup script.

## Usage

1. **Configure Your Calculation**:
   Modify the `INPUT.txt` file (or equivalent configuration) in your working directory. This file controls all USPEX parameters (population size, number of generations, fitness function, etc.).

2. **Select Fitness Calculator**:
   Ensure your `calculator` (or similar) script is correctly set up to call your chosen DFT/MD code (e.g., SIESTA, GULP, VASP, Quantum ESPRESSO) based on the generated structure files.

3. **Launch the Python package**:
   From the directory that contains `INPUT.txt` and `MOL_*`, run:
   ```bash
   gpevo
   ```
   or:
   ```bash
   python -m gpevo
   ```
   This port currently implements the `calculationType=310` molecular-crystal
   path and resolves its bundled helper files from the installed package. The
   original MATLAB/OCTAVE entry point remains available:
   ```octave
   >> USPEX
   ```

4. **Monitor & Analyze**:
   Check generation folders (`generation_X`) for output files. Use the visualization scripts in `FunctionFolder/USPEX/` to analyze results.

5. **use Gaussian process acitive learning**:

  1      : gp          % 开关（0/1），=1 时在弛豫后追加一步 GP
  24     : ncpu        % gp() 的 --n；不写则取 % numProcessors 最后一项，再不济 8
  data   : data        % 数据目录（--data）
  den    : 1.8         % 密度阈值：gp() 的 --den + CalcFitness 的 EI 门
  0.04   : u           % 主动学习不确定度阈值（超此值才送 DFT）
  2      : k           % 每代 EI top-K 个数
  ei     : mode        % ei | random | uncertainty


## File Structure

* **FunctionFolder/**: Main source code.
    * **AbinitCode/**: Interfaces and I/O for various simulation codes (VASP, QE, LAMMPS, etc.).
    * **USPEX/**: Core evolutionary algorithm modules and structure prediction logic (`.m` files).
    * **PSO/**: Particle Swarm Optimization implementation.
    * **METADYNAMICS/**: Metadynamics-related functions.
    * **sys/**: System utility functions, core Cython extensions (`.pyx`, `.c`), and Python helpers.
    * **Tool/**: External tools and scripts.
    * **spacegroup/**: Symmetry analysis tools.
* **README.md**: This file.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the [LICENSE NAME] License - see the [LICENSE](LICENSE) file for details. As an open-source project on Gitee, please ensure you comply with the specific terms of its license.

## Contact

* **Author**: Fenggo
* **Github Profile**: [https://github.com/fenggo](https://github.com/fenggo)
* **Project Home**: [https://github.com/fenggo/uspex-gp](https://github.com/fenggo/uspex-gp)

---
*Note: This README was generated based on the repository structure. Please verify specific details with the code comments and documentation within the repository.*