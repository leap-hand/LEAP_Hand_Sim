# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on:
# https://github.com/HaozhiQi/hora/blob/main/hora/algo/deploy/deploy.py
# --------------------------------------------------------


from attr import has
import isaacgym
import torch
import xml.etree.ElementTree as ET
import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import to_absolute_path
from leapsim.utils.reformat import omegaconf_to_dict, print_dict
from leapsim.utils.utils import set_np_formatting, set_seed, get_current_commit_hash
from leapsim.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner, _override_sigma, _restore
from rl_games.algos_torch import model_builder
from leapsim.learning import amp_continuous
from leapsim.learning import amp_players
from leapsim.learning import amp_models
from leapsim.learning import amp_network_builder
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from collections import deque
import math
import random

class HardwarePlayer(object):
    def __init__(self, config):
        self.config = omegaconf_to_dict(config)
        self.set_defaults()
        self.action_scale = 1 / 24
        self.actions_num = 16
        self.device = 'cuda'

        self.debug_viz = self.config["task"]['env']['enableDebugVis']

        # hand setting
        self.init_pose = self.fetch_grasp_state()
        self.get_dof_limits()
        # self.leap_dof_lower = torch.from_numpy(np.array([
        #     -1.5716, -0.4416, -1.2216, -1.3416,  1.0192,  0.0716,  0.2516, -1.3416,
        #     -1.5716, -0.4416, -1.2216, -1.3416, -1.5716, -0.4416, -1.2216, -1.3416
        # ])).to(self.device)
        # self.leap_dof_upper = torch.from_numpy(np.array([
        #     1.5584, 1.8584, 1.8584, 1.8584, 1.7408, 1.0684, 1.8584, 1.8584, 1.5584,
        #     1.8584, 1.8584, 1.8584, 1.5584, 1.8584, 1.8584, 1.8584
        # ])).to(self.device)

        # Modifers to remove conservative clipping
        # self.leap_dof_lower[4] = -0.519205
        # self.leap_dof_upper[5] = 1.96841
        # self.leap_dof_lower[5] = -0.57159
        # self.leap_dof_lower[6] = -0.25159

        if self.debug_viz:
            self.setup_plot()

    def real_to_sim(self, values):
        if not hasattr(self, "real_to_sim_indices"):
            self.construct_sim_to_real_transformation()

        return values[:, self.real_to_sim_indices]

    def sim_to_real(self, values):
        if not hasattr(self, "sim_to_real_indices"):
            self.construct_sim_to_real_transformation()
        
        return values[:, self.sim_to_real_indices]

    def construct_sim_to_real_transformation(self):
        self.sim_to_real_indices = self.config["task"]["env"]["sim_to_real_indices"]
        self.real_to_sim_indices= self.config["task"]["env"]["real_to_sim_indices"]

    def get_dof_limits(self):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
        hand_asset_file = self.config['task']['env']['asset']['handAsset']

        tree = ET.parse(os.path.join(asset_root, hand_asset_file))
        root = tree.getroot()

        self.leap_dof_lower = [0 for _ in range(16)]
        self.leap_dof_upper = [0 for _ in range(16)]

        for child in root.getchildren():
            if child.tag == "joint":
                joint_idx = int(child.attrib['name'])

                for gchild in child.getchildren():
                    if gchild.tag == "limit":
                        lower = float(gchild.attrib['lower'])
                        upper = float(gchild.attrib['upper'])

                        self.leap_dof_lower[joint_idx] = lower
                        self.leap_dof_upper[joint_idx] = upper

        self.leap_dof_lower = torch.tensor(self.leap_dof_lower).to(self.device)[None, :] 
        self.leap_dof_upper = torch.tensor(self.leap_dof_upper).to(self.device)[None, :] 

        self.leap_dof_lower = self.real_to_sim(self.leap_dof_lower).squeeze()
        self.leap_dof_upper = self.real_to_sim(self.leap_dof_upper).squeeze()

    def plot_callback(self):
        self.fig.canvas.restore_region(self.bg)

        # self.ydata.append(self.object_rpy[0, 2].item())
        self.ydata.append(self.cur_obs_joint_angles[0, 9].item())
        self.ydata2.append(self.cur_obs_joint_angles[0, 4].item())

        self.ln.set_ydata(list(self.ydata))
        self.ln.set_xdata(range(len(self.ydata)))

        self.ln2.set_ydata(list(self.ydata2))
        self.ln2.set_xdata(range(len(self.ydata2)))

        self.ax.draw_artist(self.ln)
        self.ax.draw_artist(self.ln2)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()
    
    def setup_plot(self):   
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-1, 1)
        self.ydata = deque(maxlen=100) # Plot 5 seconds of data 
        self.ydata2 = deque(maxlen=100)
        (self.ln,) = self.ax.plot(range(len(self.ydata)), list(self.ydata), animated=True)
        (self.ln2,) = self.ax.plot(range(len(self.ydata2)), list(self.ydata2), animated=True)
        plt.show(block=False)
        plt.pause(0.1)

        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.ln)
        self.fig.canvas.blit(self.fig.bbox)

    def set_defaults(self):
        if "include_history" not in self.config["task"]["env"]:
            self.config["task"]["env"]["include_history"] = True

        if "include_targets" not in self.config["task"]["env"]:
            self.config["task"]["env"]["include_targets"] = True

    def fetch_grasp_state(self, s=1.0):
        self.grasp_cache_name = self.config['task']['env']['grasp_cache_name']
        grasping_states = np.load(f'cache/{self.grasp_cache_name}_grasp_50k_s{str(s).replace(".", "")}.npy')

        if "sampled_pose_idx" in self.config["task"]["env"]:
            idx = self.config["task"]["env"]["sampled_pose_idx"]
        else:
            idx = random.randint(0, grasping_states.shape[0] - 1)

        return grasping_states[idx][:16] # first 16 are hand dofs, last 16 is object state

    def deploy(self):
        import rospy
        from hardware_controller import LeapHand
        
        # try to set up rospy
        num_obs = self.config['task']['env']['numObservations'] 
        num_obs_single = num_obs // 3 
        rospy.init_node('example')
        leap = LeapHand()
        leap.leap_dof_lower = self.leap_dof_lower.cpu().numpy()
        leap.leap_dof_upper = self.leap_dof_upper.cpu().numpy()
        leap.sim_to_real_indices = self.sim_to_real_indices
        leap.real_to_sim_indices = self.real_to_sim_indices
        # Wait for connections.
        rospy.wait_for_service('/leap_position')

        hz = 20
        self.control_dt = 1 / hz
        ros_rate = rospy.Rate(hz)

        print("command to the initial position")
        for _ in range(hz * 4):
            leap.command_joint_position(self.init_pose)
            obses, _ = leap.poll_joint_position()
            ros_rate.sleep()
        print("done")

        obses, _ = leap.poll_joint_position()

        # hardware deployment buffer
        obs_buf = torch.from_numpy(np.zeros((1, 0)).astype(np.float32)).cuda()

        def unscale(x, lower, upper):
            return (2.0 * x - upper - lower) / (upper - lower)

        obses = torch.from_numpy(obses.astype(np.float32)).cuda()
        prev_target = obses[None].clone()
        cur_obs_buf = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)[None]

        if self.config["task"]["env"]["include_history"]:
            num_append_iters = 3
        else:
            num_append_iters = 1

        for i in range(num_append_iters):   
            obs_buf = torch.cat([obs_buf, cur_obs_buf.clone()], dim=-1)
            
            if self.config["task"]["env"]["include_targets"]:
                obs_buf = torch.cat([obs_buf, prev_target.clone()], dim=-1)

            if "phase_period" in self.config["task"]["env"]:
                phase = torch.tensor([[0., 1.]], device=self.device)
                obs_buf = torch.cat([obs_buf, phase], dim=-1)

        if "obs_mask" in self.config["task"]["env"]:
            obs_buf = obs_buf * torch.tensor(self.config["task"]["env"]["obs_mask"]).cuda()[None, :]

        obs_buf = obs_buf.float()

        counter = 0 

        if "debug" in self.config["task"]["env"]:
            self.obs_list = []
            self.target_list = []

            if "record" in self.config["task"]["env"]["debug"]:
                self.record_duration = int(self.config["task"]["env"]["debug"]["record"]["duration"] / self.control_dt)

            if "actions_file" in self.config["task"]["env"]["debug"]:
                self.actions_list = torch.from_numpy(np.load(self.config["task"]["env"]["debug"]["actions_file"])).cuda()        
                self.record_duration = self.actions_list.shape[0]

        if self.player.is_rnn:
            self.player.init_rnn()

        while True:
            counter += 1
            # obs = self.running_mean_std(obs_buf.clone()) # ! Need to check if this is implemented
            
            if hasattr(self, "actions_list"):
                action = self.actions_list[counter-1][None, :]
            else:
                action = self.forward_network(obs_buf)

            action = torch.clamp(action, -1.0, 1.0)

            if "actions_mask" in self.config["task"]["env"]:
                action = action * torch.tensor(self.config["task"]["env"]["actions_mask"]).cuda()[None, :]

            target = prev_target + self.action_scale * action 
            target = torch.clip(target, self.leap_dof_lower, self.leap_dof_upper)
            prev_target = target.clone()
        
            # interact with the hardware
            commands = target.cpu().numpy()[0]

            if "disable_actions" not in self.config["task"]["env"]:
                leap.command_joint_position(commands)

            ros_rate.sleep()  # keep 20 Hz command
            
            # command_list.append(commands)
            # get o_{t+1}
            obses, _ = leap.poll_joint_position()
            obses = torch.from_numpy(obses.astype(np.float32)).cuda()

            # obs_buf_list.append(obses.cpu().numpy().squeeze())
            cur_obs_buf = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)[None]
            self.cur_obs_joint_angles = cur_obs_buf.clone()

            if self.debug_viz:
                self.plot_callback()

            if hasattr(self, "obs_list"):
                self.obs_list.append(cur_obs_buf[0].clone())
                self.target_list.append(target[0].clone().squeeze())

                if counter == self.record_duration - 1:
                    self.obs_list = torch.stack(self.obs_list, dim=0)
                    self.obs_list = self.obs_list.cpu().numpy()

                    self.target_list = torch.stack(self.target_list, dim=0)
                    self.target_list = self.target_list.cpu().numpy()

                    if "actions_file" in self.config["task"]["env"]["debug"]:
                        actions_file = os.path.basename(self.config["task"]["env"]["debug"]["actions_file"])
                        folder = os.path.dirname(self.config["task"]["env"]["debug"]["actions_file"])
                        suffix = "_".join(actions_file.split("_")[1:])
                        joints_file = os.path.join(folder, "joints_real_{}".format(suffix)) 
                        target_file = os.path.join(folder, "targets_real_{}".format(suffix))
                    else:
                        suffix = self.config["task"]["env"]["debug"]["record"]["suffix"]
                        joints_file = "debug/joints_real_{}.npy".format(suffix)
                        target_file = "debug/targets_real_{}.npy".format(suffix)

                    np.save(joints_file, self.obs_list)
                    np.save(target_file, self.target_list) 
                    exit()

            if self.config["task"]["env"]["include_history"]:
                obs_buf = obs_buf[:, num_obs_single:].clone()
            else:
                obs_buf = torch.zeros((1, 0), device=self.device)

            obs_buf = torch.cat([obs_buf, cur_obs_buf.clone()], dim=-1)

            if self.config["task"]["env"]["include_targets"]:
                obs_buf = torch.cat([obs_buf, target.clone()], dim=-1)

            if "phase_period" in self.config["task"]["env"]:
                omega = 2 * math.pi / self.config["task"]["env"]["phase_period"]
                phase_angle = (counter - 1) * omega / hz 
                num_envs = obs_buf.shape[0]
                phase = torch.zeros((num_envs, 2), device=obs_buf.device)
                phase[:, 0] = math.sin(phase_angle)
                phase[:, 1] = math.cos(phase_angle)
                obs_buf = torch.cat([obs_buf, phase.clone()], dim=-1)

            if "obs_mask" in self.config["task"]["env"]:
                obs_buf = obs_buf * torch.tensor(self.config["task"]["env"]["obs_mask"]).cuda()[None, :]

            obs_buf = obs_buf.float()

    def forward_network(self, obs):
        return self.player.get_action(obs, True)

    def restore(self):
        rlg_config_dict = self.config['train']
        rlg_config_dict["params"]["config"]["env_info"] = {}
        self.num_obs = self.config["task"]["env"]["numObservations"]
        self.num_actions = 16
        observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        rlg_config_dict["params"]["config"]["env_info"]["observation_space"] = observation_space
        action_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)
        rlg_config_dict["params"]["config"]["env_info"]["action_space"] = action_space
        rlg_config_dict["params"]["config"]["env_info"]["agents"] = 1

        def build_runner(algo_observer):
            runner = Runner(algo_observer)
            runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
            runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
            model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
            model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

            return runner

        runner = build_runner(RLGPUAlgoObserver())
        runner.load(rlg_config_dict)
        runner.reset()

        args = {
            'train': False,
            'play': True,
            'checkpoint' : self.config['checkpoint'],
            'sigma' : None
        }

        self.player = runner.create_player()
        _restore(self.player, args)
        _override_sigma(self.player, args)
        

@hydra.main(config_name='config', config_path='cfg')
def main(config: DictConfig):
    agent = HardwarePlayer(config)
    agent.restore()
    agent.deploy()

if __name__ == '__main__':
    main()