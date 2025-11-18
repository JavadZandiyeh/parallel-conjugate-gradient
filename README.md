# parallel-conjugate-gradient
UniTrento HPC4DS Course Project: Parallel Conjugate Gradient

# Build instructions
### Prerequisites
To build the project from source, ensure you have the following prerequisites installed:
- C++ compiler with MPI support (e.g., mpicxx)
- MPI library (e.g., OpenMPI, MPICH)

Then, create a Python virtual environment and install the required packages:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install the `petsc` library following the instructions at [PETSc Installation Guide](https://petsc.org/release/install/). An script for a cluster installation is provided in `scripts/install_petsc.sh`.
Set the `PETSC_DIR` variable to point to your PETSc installation directory.
```bash
export PETSC_DIR=/path/to/petsc
```
Set the `PETSC_DIR` variable also in the `build.yaml` file.

### Dataset Preparation
To download the datasets required for testing the parallel conjugate gradient implementation, run the following command:
```bash
mtxman sync scripts/datasets.yaml
```
This will download the matrices specified in the YAML file into the `datasets` directory.

The downloaded matrices have to be converted to PETSc binary format. You can use the provided script to perform this conversion:
```bash
scripts/convert_datasets.sh
```

### Building the Project
Before building the project, initialize sbatchman in the project directory if you haven't done so already:
```bash
sbatchman init
```
If you are running on the university cluster use `hpc_unitn` as the cluster name. Use the following command to load the cluster configuration:
```bash
sbatchman configure scripts/config.yaml
```

To build the project, run the following command:
```bash
sbatchman launch build.yaml
```

### Running the Project