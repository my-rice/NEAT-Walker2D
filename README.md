# NEAT-Walker2D

# Description
This repository contains the code for training a NEAT agent to solve the Walker2D environment from the OpenAI Gym. The code is based on the NEAT-Python library, but it was modified to work with mpi4py, which allows the code to run in parallel, in particular, each process runs a different population. At the end of the specified number of generations, the best agent from each population is saved to a file and shared with the near processes (this idea is named "migration" in the project). Note that at the last migration step the best agent is shared with all the processes. The code also uses the Hydra library to manage the configuration parameters. 

# Project goal
The goal of this project is to use a famous NeuroEvolution algorithm, known as NEAT, to make the Walker2D agent move forward at 1 m/s speed, and achieving stability and efficiency in the movement.  

# Final result video:
![alt text](https://github.com/my-rice/NEAT-Walker2D/blob/main/video/image88.gif "Logo Title Text 1")


# How to run the training code

1. Install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

To install `mpi4py`, you can follow the instructions in this [link](https://mpi4py.readthedocs.io/en/stable/install.html)

2. Run the training code by running the following command:
   
```bash
mpiexec --oversubscribe -np <number of processes> python3 main.py <hydra_configuration_paramters>
```

For example:

```bash
mpiexec --oversubscribe -np 9 python3 main.py experiment_name=experiment_1
```

# How to run the evaluation code

1. Install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

2. Run the evaluation code by running the following command:

```bash
python3 evaluate_agent.py <path_to_model>
```

or

```bash
python3 evaluate_agent.py <path_to_dir_containing_models>
```

For example:

```bash
python3 evaluate_agent.py results/STANDARD2-gen/best_net_1.pkl
```

## Common problems

### How to install Mujoco for this environment

1. Download the Mujoco library from this [link](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz)

2. Create a hidden folder:

```bash
mkdir /home/username/.mujoco
```

3. Extract the library to the `.mujoco` folder.

```bash
tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
```

4. Include these lines in `.bashrc` file replacing username with the real username:

```bash
echo -e 'export LD_LIBRARY_PATH=/home/user-name/.mujoco/mujoco210/bin 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia 
export PATH="$LD_LIBRARY_PATH:$PATH" 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so' >> ~/.bashrc
```

5. Source `.bashrc`.

```bash
source ~/.bashrc
```

6. Install the following version of gymnasium

```bash
pip install gymnasium==1.0.0a1
```
