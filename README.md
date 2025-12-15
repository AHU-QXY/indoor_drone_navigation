# Deep Reinforcement Learning for UAV Navigation in Indoor_Drone_Navigation Environments
This repository provides an implementation of our deep reinforcement learning approach for autonomous UAV navigation in indoor_drone_navigation environments. Our method combines path planning with DRL to help drones achieve a series of sub target tasks through task decomposition, ultimately completing global navigation tasks. This technique significantly improves navigation performance compared to standard DRL algorithms.
### 🔗 **Dependencies**
This experiment is conducted within the XuanCe framework and utilizes the gym-pybullet-drones UAV simulation engine for drone simulation.
- [XuanCe Framework](https://github.com/agi-brain/xuance) - A reinforcement learning framework used for algorithm implementation.
- [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) - A drone simulation environment built on PyBullet.

---
## Requirements
Open terminal and type the following commands, then a new conda environment for xuance with drones could be built:
```
conda create -n xuance_drones python=3.10
conda activate xuance_drones
pip install xuance  

git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones/
pip install --upgrade pip
pip install -e .  # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`
```
After the installation ends you can activate your environment with:
```
source activate xuance_drones
```
## Instructions 

### Conduct task training

#### DDPG Example
To initiate training using the DDPG algorithm, execute the following command:

```
python ddpg.py --device "cuda:0" --test 0 --seed 0 
```
#### SAC Example
To initiate training using the SAC algorithm, execute the following command:

```
python sac.py --device "cuda:0" --test 0 --seed 0 
```
When the training is complete, the data and lines can be observed in tensorboard.


```
tensorboard --logdir logs
```
### Test the trained model
```
python ddpg.py --seed 0 --test 1 --test_episode 1 --model_folder your_model_folder
```
or
```
python sac.py --seed 0 --test 1 --test_episode 1 --model_folder your_model_folder
```
