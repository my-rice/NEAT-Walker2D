# legged_locomotion
Natural Computation project

# How to run the training code
1. Install the required packages by running the following command:
```bash
pip install -r requirements.txt
```

2. Run the training code by running the following command:
```bash
mpiexec --oversubscribe -np <number of processes> python3 main.py <hydra_configuration_paramters>
```
A real 



## Common problem: How to install Mujoco for this environment
Download the Mujoco library from this link: https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz

*Create a hidden folder :*
mkdir /home/username/.mujoco

*Extract the library to the .mujoco folder.*
tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/

*Include these lines in .bashrc file replacing username with the real username:*
echo -e 'export LD_LIBRARY_PATH=/home/user-name/.mujoco/mujoco210/bin 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia 
export PATH="$LD_LIBRARY_PATH:$PATH" 
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so' >> ~/.bashrc

*Source bashrc.*
source ~/.bashrc

*Install this version of gymnasium*
pip install gymnasium==1.0.0a1
