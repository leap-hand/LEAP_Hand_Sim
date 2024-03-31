# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand
This repository contains the URDF, IsaacGym environment and sim2real deployment code for the paper "LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning" ([https://arxiv.org/abs/2309.06440](https://arxiv.org/abs/2309.06440)).  

## Installation
 
Setup a conda environment (optional)

```
conda create -n leapsim python=3.8
conda activate leapsim
```
Install Pytorch using [these instructions](https://pytorch.org/get-started/locally/)

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions in the documentation  
```
cd isaacgym/python
pip install -e .
```
Clone and install leapsim python packages
```
git clone https://github.com/leap-hand/LEAP_Hand_Sim
cd LEAP_Hand_Sim
pip install matplotlib gitpython numpy==1.20.3 wandb
pip install -e .
```
## Running a pretrained policy
You can run a pretrained in-hand reorienation policy to check your install. To deploy this policy on the real hand, see the real-world deployment section below. 
```
cd leapsim
python3 train.py wandb_activate=false num_envs=1 headless=false test=true task=LeapHandRot checkpoint=runs/pretrained/nn/LeapHand.pth
```
![sim-deployment](docs/images/sim.gif)

## Real-world deployment
- Running in the real world requires our [LEAP Hand ROS API](https://github.com/leap-hand/LEAP_Hand_API/tree/main/ros_module).
- Follow the instructions in the above link and then run ```roslaunch example.launch``` first.  The hand should go to the home pose.
- Next, in a separate window run deploy.py using:
```
cd leapsim
python3 deploy.py wandb_activate=false num_envs=1 headless=false test=true task=LeapHandRot checkpoint=runs/pretrained/nn/LeapHand.pth
```
- The hand should go to a pre-grasp pose and then rotate a 7.5cm cube by default.

![rw-deployment](docs/images/rw.gif)

## Training your own policy
First, generate a cache of stable grasps for different cube sizes

```
for cube_scale in 0.9 0.95 1.0 1.05 1.1 
do
	bash scripts/gen_grasp.sh $cube_scale custom_grasp_cache num_envs=1024 
done
```

This will generate `.npy` files in the `leapsim/cache` folder. Next, train a policy using this grasp cache

```
python3 train.py task=LeapHandRot max_iterations=1000 task.env.grasp_cache_name=custom_grasp_cache
```

If you wish to not use wandb append `wandb_activate=false`

```
for cube_scale in 0.9 0.95 1.0 1.05 1.1 
do
	bash scripts/gen_grasp.sh $cube_scale custom_grasp_cache num_envs=1024 wandb_activate=false
done
python3 train.py task=LeapHandRot max_iterations=1000 task.env.grasp_cache_name=custom_grasp_cache wandb_activate=false
```

After training, the policy can be visualized by running

```
python3 train.py wandb_activate=false num_envs=1 headless=false test=true task=LeapHandRot checkpoint=runs/<checkpoint_name>/nn/LeapHand.pth
```
For training details of this policy refer to the LEAP hand [paper](https://arxiv.org/abs/2309.06440) section VI-D. 

## Citing
If you find LEAP hand or this codebase useful in your research, please cite: 
```
@article{
	shaw2023leaphand,
	title={LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning},
	author={Shaw, Kenneth and Agarwal, Ananye and Pathak, Deepak},
	journal={Robotics: Science and Systems (RSS)},
	year={2023}
}
```

## Acknowledgements

Check out the following amazing codebases we build upon
- Hora - [https://github.com/HaozhiQi/hora/](https://github.com/HaozhiQi/hora/)
- IsaacGymEnvs - [https://github.com/NVIDIA-Omniverse/IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
